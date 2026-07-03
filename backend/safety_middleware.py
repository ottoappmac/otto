"""Runtime safety guards that move correctness rules out of the prompt.

Four guards live here, all implemented as ``wrap_tool_call`` interceptors:

1. **Execute path safety** (:class:`ExecutePathSafetyMiddleware`)

   Rewrites ``execute`` commands that reference bare ``/output/...`` (or any
   absolute path that resolves to the host root and isn't under the
   session sandbox) to use ``$SESSION_FILES/...`` instead.  This is the
   highest-risk prompt-only rule today (``R9`` in
   ``src/deep_agent/prompt.py``); promoting it to runtime means lite-mode
   prompts can drop the long-form explanation without losing the
   guarantee.

2. **Subagent-as-tool guard** (:class:`SubagentAsToolGuardMiddleware`)

   When the model emits a tool call whose name matches a registered
   library subagent (``web-voyager``, ``computer-voyager``, …), the call
   is rewritten in-flight into the correct
   ``task(subagent_type="<name>", description="...")`` shape instead of
   failing with a ``ToolNode`` validation error.  This is the single
   most common dispatch failure on smaller open-source models.

3. **High-risk command flagging** (:class:`HighRiskExecuteFlaggerMiddleware`)

   Before an ``execute`` call hits the human-in-the-loop interrupt, the
   command is screened for known-dangerous patterns (``rm -rf /``,
   ``git push --force``, raw block-device writes, …).  Matches are
   logged at WARNING and a ``high_risk=True`` marker is added to the
   tool-call ``args`` so the frontend can render a high-prominence
   approval badge.  The interrupt continues to gate execution; this
   middleware only adds visibility.

4. **Live output streaming** (:class:`LiveOutputMiddleware`)

   Wraps the ``execute`` command with ``tee`` so stdout/stderr flow to
   both the deepagents backend (unchanged tool semantics) and a
   per-session :class:`asyncio.Queue`.  The queue is drained by
   :meth:`~backend.session_manager.SessionManager._do_stream` and
   forwarded to the frontend as ``execute_output`` WebSocket events,
   enabling a live scrolling tail while the command runs.  This
   middleware fires **after** the HITL interrupt (the approval dialog
   shows the original command; the tee wrapper is added transparently).

All four guards are independent and additive — they compose with each
other and with :class:`PatchToolCallsMiddleware`,
:class:`SummarizationMiddleware`, etc., in the standard order.  They are
registered by :mod:`backend.session_manager` for both the orchestrator
and library subagents.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────


def _tool_call_name(request: ToolCallRequest) -> str:
    return request.tool_call.get("name", "") or ""


def _tool_call_args(request: ToolCallRequest) -> dict:
    args = request.tool_call.get("args") or {}
    return args if isinstance(args, dict) else {}


def _quote_context(command: str, idx: int) -> Optional[str]:
    """Return the shell quoting context at offset ``idx`` in ``command``.

    Walks ``command[:idx]`` tracking single- and double-quote state (POSIX
    shell rules: a backslash escapes the next char only inside double quotes
    or when unquoted; single quotes are literal and ignore backslashes).

    Returns ``"single"`` if ``idx`` sits inside a single-quoted span,
    ``"double"`` if inside a double-quoted span, or ``None`` if unquoted.
    """
    in_single = False
    in_double = False
    i = 0
    n = len(command)
    while i < idx and i < n:
        ch = command[i]
        if in_single:
            if ch == "'":
                in_single = False
        elif in_double:
            if ch == "\\" and i + 1 < n:
                # Backslash escapes the next char inside double quotes.
                i += 2
                continue
            if ch == '"':
                in_double = False
        else:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == "'":
                in_single = True
            elif ch == '"':
                in_double = True
        i += 1
    if in_single:
        return "single"
    if in_double:
        return "double"
    return None


# ── 1. Execute path safety ───────────────────────────────────────────────


# Matches absolute paths beginning with /output, /tmp-style session-internal
# directories that the orchestrator sometimes confuses with the virtual
# filesystem.  We intentionally do NOT rewrite arbitrary absolute paths
# (``/etc/...``, ``/usr/...``) because some shell commands legitimately
# reference them.
_BARE_OUTPUT_RE = re.compile(r"(?<![A-Za-z0-9_/\-\.])(/output(?:/[^\s'\"`)]+)?)")

# Advisory appended to an ``execute`` result when the command embeds file
# paths that the command-string rewrite cannot safely fix (paths inside a
# heredoc / ``-c`` script body, or a literal ``$SESSION_FILES`` that the
# shell will not expand), or when the result itself shows a path error.  It
# steers the model to the one correct pattern instead of silently writing
# outside the session dir or looping on the failure.
_PATH_ADVISORY = (
    "\n\n[path-safety] If a script run via execute reads/writes files: do NOT "
    "hardcode '/output/...' (that resolves to the host filesystem root, not your "
    "session) and do NOT use the literal string '$SESSION_FILES' inside Python "
    "source (it is not shell-expanded there). Instead use "
    "os.environ['SESSION_FILES'] — e.g. "
    "os.path.join(os.environ['SESSION_FILES'], 'output', 'file.html') — or a path "
    "relative to the working directory (e.g. 'output/file.html'), since execute "
    "already runs in the session root. After writing, confirm the file exists with "
    "read_file('/output/file.html')."
)


def _has_inline_script_paths(command: str) -> bool:
    """True if ``command`` embeds a risky path inside a heredoc or ``-c`` body.

    The command-string rewrite (``_BARE_OUTPUT_RE``) can turn ``/output`` into
    ``"$SESSION_FILES/output"`` for plain shell tokens, but when ``/output`` or
    ``$SESSION_FILES`` lives inside a Python/JS script body (``python3 << EOF``
    or ``python -c '...'``) that rewrite is either ineffective or actively wrong
    (a literal ``$SESSION_FILES`` ends up in the script source). Detect those so
    we can advise instead.
    """
    risky = ("/output" in command) or ("$SESSION_FILES" in command)
    if not risky:
        return False
    if "<<" in command:  # heredoc
        return True
    if re.search(r"(?<![A-Za-z0-9_])-c\b", command):  # python/node -c "..."
        return True
    return False


def _has_literal_session_files(command: str) -> bool:
    """True if ``$SESSION_FILES`` appears single-quoted (won't shell-expand)."""
    for m in re.finditer(r"\$SESSION_FILES", command):
        if _quote_context(command, m.start()) == "single":
            return True
    return False


def _result_shows_path_error(text: str) -> bool:
    """True if a tool result looks like a path error involving our anchors."""
    if not text:
        return False
    has_err = ("No such file" in text) or ("FileNotFoundError" in text)
    has_path = ("$SESSION_FILES" in text) or ("/output" in text)
    return has_err and has_path


class ExecutePathSafetyMiddleware(AgentMiddleware):
    """Rewrite bare ``/output/...`` references in ``execute`` commands.

    The session's filesystem sandbox lives under ``$SESSION_FILES``, but
    the file tools (``write_file`` / ``read_file`` / ``ls``) treat ``/``
    as a virtual root.  When the orchestrator carries that mental model
    over to ``execute``, ``/output/foo.py`` resolves to the *host*
    filesystem root — leaking writes outside the session.

    This middleware matches every ``/output/...`` substring inside the
    ``command`` argument and rewrites it to ``$SESSION_FILES/output/...``.
    Other absolute paths are left alone.

    The emitted form is quote-context-aware so the rewrite survives session
    paths that contain spaces (e.g. macOS ``~/Library/Application
    Support/...``).  ``execute`` runs via ``subprocess.run(shell=True)`` —
    i.e. ``/bin/sh`` — which word-splits an unquoted ``$SESSION_FILES`` on
    that space.  Depending on the quoting context of the matched token we
    emit:

    * unquoted     -> ``"$SESSION_FILES/output/..."`` (self-contained,
      quotes prevent word-splitting; ``"$VAR"/literal`` is valid POSIX)
    * double-quoted -> ``$SESSION_FILES/output/...`` (already inside double
      quotes, so the variable expands and the space is preserved)
    * single-quoted -> ``'"$SESSION_FILES/output/..."'`` (close the single
      quote, switch to double quotes so the variable actually expands, then
      reopen the single quote)

    Known limitation: a path with an *internal* space that the model wrapped
    in quotes (e.g. ``"/output/a b.txt"``) is still mishandled, because
    ``_BARE_OUTPUT_RE`` stops the matched token at whitespace.  This is
    pre-existing behaviour and out of scope here.
    """

    def __init__(self, *, tool_name: str = "execute") -> None:
        self._tool_name = tool_name

    def _rewrite(self, command: str) -> tuple[str, bool]:
        if "/output" not in command:
            return command, False

        rewrote = False

        def _sub(match: re.Match) -> str:
            nonlocal rewrote
            tail = match.group(1)
            rewrote = True
            context = _quote_context(command, match.start(1))
            if context == "double":
                return "$SESSION_FILES" + tail
            if context == "single":
                return "'\"$SESSION_FILES" + tail + "\"'"
            return '"$SESSION_FILES' + tail + '"'

        new_cmd = _BARE_OUTPUT_RE.sub(_sub, command)
        return new_cmd, rewrote

    def _maybe_rewrite_request(self, request: ToolCallRequest) -> ToolCallRequest:
        if _tool_call_name(request) != self._tool_name:
            return request
        args = _tool_call_args(request)
        cmd = args.get("command")
        if not isinstance(cmd, str):
            return request
        new_cmd, changed = self._rewrite(cmd)
        if not changed:
            return request
        logger.warning(
            "ExecutePathSafetyMiddleware: rewrote bare /output path in execute command "
            "(call_id=%s)\n  before: %s\n  after:  %s",
            request.tool_call.get("id"), cmd, new_cmd,
        )
        new_call = {**request.tool_call, "args": {**args, "command": new_cmd}}
        return request.override(tool_call=new_call)

    def _original_command(self, request: ToolCallRequest) -> Optional[str]:
        if _tool_call_name(request) != self._tool_name:
            return None
        cmd = _tool_call_args(request).get("command")
        return cmd if isinstance(cmd, str) else None

    def _maybe_append_advisory(
        self,
        original_cmd: Optional[str],
        result: ToolMessage | Command[Any],
    ) -> ToolMessage | Command[Any]:
        """Append the path-safety advisory to an execute result when warranted."""
        if not isinstance(result, ToolMessage):
            return result

        advise = False
        if isinstance(original_cmd, str) and (
            _has_inline_script_paths(original_cmd)
            or _has_literal_session_files(original_cmd)
        ):
            advise = True

        content = result.content
        text = content if isinstance(content, str) else str(content)
        if _result_shows_path_error(text):
            advise = True

        if not advise or not isinstance(content, str):
            return result
        if "[path-safety]" in content:  # avoid double-appending
            return result

        logger.warning(
            "ExecutePathSafetyMiddleware: appended path-safety advisory to execute "
            "result (risky inline path or path error detected)"
        )
        return result.model_copy(update={"content": content + _PATH_ADVISORY})

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        original_cmd = self._original_command(request)
        result = handler(self._maybe_rewrite_request(request))
        return self._maybe_append_advisory(original_cmd, result)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        original_cmd = self._original_command(request)
        result = await handler(self._maybe_rewrite_request(request))
        return self._maybe_append_advisory(original_cmd, result)


# ── 2. Subagent-as-tool guard ────────────────────────────────────────────


class SubagentAsToolGuardMiddleware(AgentMiddleware):
    """Rewrite ``<subagent>(args)`` calls into the correct ``task(...)`` form.

    The orchestrator can delegate to library subagents (``web-voyager``,
    ``computer-voyager``, …) only via the ``task`` tool — they are not
    bound as tools themselves.  Smaller open-source models routinely
    emit ``web_voyager(prompt="…")`` directly, which fails validation in
    ``ToolNode``.  Rather than letting that fail and forcing a recovery
    turn, we intercept the call here and rewrite it in flight.

    Matching is permissive (kebab-case, snake_case, and CamelCase are
    all accepted) because models normalize names inconsistently.
    """

    _TASK_TOOL = "task"
    _DESCRIPTION_KEYS = ("description", "prompt", "task", "instructions", "input")

    def __init__(self, subagent_names: Iterable[str]) -> None:
        cleaned = [n for n in (s.strip() for s in subagent_names) if n]
        self._exact = set(cleaned)
        self._normalized = {self._normalize(n): n for n in cleaned}

    @staticmethod
    def _normalize(name: str) -> str:
        return re.sub(r"[\s_\-]+", "", name).lower()

    def _resolve_subagent(self, tool_name: str) -> str | None:
        if not tool_name:
            return None
        if tool_name in self._exact:
            return tool_name
        return self._normalized.get(self._normalize(tool_name))

    def _maybe_rewrite_request(self, request: ToolCallRequest) -> ToolCallRequest:
        called = _tool_call_name(request)
        if called == self._TASK_TOOL:
            return request
        target = self._resolve_subagent(called)
        if target is None:
            return request

        args = _tool_call_args(request)
        description: str | None = None
        for key in self._DESCRIPTION_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                description = val
                break
        if description is None and args:
            description = "; ".join(f"{k}={v}" for k, v in args.items())
        if not description:
            description = "(no description provided)"

        new_call = {
            **request.tool_call,
            "name": self._TASK_TOOL,
            "args": {"subagent_type": target, "description": description},
        }
        logger.warning(
            "SubagentAsToolGuardMiddleware: rewrote %r call into task(subagent_type=%r) "
            "(call_id=%s)",
            called, target, request.tool_call.get("id"),
        )
        return request.override(tool_call=new_call)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(self._maybe_rewrite_request(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        return await handler(self._maybe_rewrite_request(request))


# ── 3. High-risk command flagging ────────────────────────────────────────


# Patterns that warrant a "high risk" badge in the approval UI.  These
# are not blocked outright — the human-in-the-loop interrupt
# (``interrupt_on={"execute": True}``) still gates execution.  We only
# raise visibility so a user clicking through approvals sees the danger.
_HIGH_RISK_PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    # ``rm -rf /`` / ``rm -rf ~`` / ``rm -rf $HOME`` and variants.
    # Trailing target may be followed by whitespace, EOL, ``/``, or the
    # end of the string — ``\b`` doesn't match before ``~`` or ``$`` so
    # we use an explicit lookahead.
    ("rm-rf-root",         re.compile(r"\brm\s+(?:-[a-zA-Z]*\s+)*(?:-rf|-fr)\s+(?:/|~|\$HOME)(?=$|\s|/)")),
    ("rm-rf-flag-split",   re.compile(r"\brm\s+(?:-[a-zA-Z]*r[a-zA-Z]*\s+)(?:-[a-zA-Z]*f[a-zA-Z]*\s+)(?:/|~|\$HOME)(?=$|\s|/)")),
    ("force-push",         re.compile(r"\bgit\s+push\s+(?:--force|-f)\b")),
    ("force-push-main",    re.compile(r"\bgit\s+push\s+.*--force.*\b(main|master)\b", re.IGNORECASE)),
    ("hard-reset",         re.compile(r"\bgit\s+reset\s+--hard\b")),
    ("dd-of-device",       re.compile(r"\bdd\b[^\n]*\bof=/dev/")),
    ("mkfs",               re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b")),
    ("chmod-recursive-/",  re.compile(r"\bchmod\s+-R\s+[0-7]+\s+/\s")),
    ("curl-pipe-shell",    re.compile(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:sh|bash|zsh)\b")),
    ("sudo-rm",            re.compile(r"\bsudo\s+rm\b")),
    ("shutdown",           re.compile(r"\b(?:shutdown|reboot|halt|poweroff)\b")),
)


def screen_high_risk_command(command: str | None) -> list[str]:
    """Return the list of risk labels matched by *command* (empty if none).

    Exposed at module level so backend routes (e.g. session approve
    endpoints) and tests can re-use the same screen the middleware does.
    The frontend mirrors this list in
    ``app/src/lib/highRiskCommands.ts`` for rendering the approval
    badge — keep the two in sync when adding a pattern.
    """
    if not command or not isinstance(command, str):
        return []
    return [label for label, pat in _HIGH_RISK_PATTERNS if pat.search(command)]


class HighRiskExecuteFlaggerMiddleware(AgentMiddleware):
    """Log ``execute`` calls that match dangerous patterns.

    Pure observability — the middleware does NOT mutate the tool call
    (which would risk breaking the tool's argument schema) and does NOT
    block (the human-in-the-loop interrupt still gates execution).  It
    emits a WARNING log line so operators reviewing logs can spot
    high-risk commands, and the frontend independently runs the same
    regex list (see :func:`screen_high_risk_command`) to render a
    high-prominence badge in the approval UI.
    """

    def __init__(self, *, tool_name: str = "execute") -> None:
        self._tool_name = tool_name

    def _log_if_risky(self, request: ToolCallRequest) -> None:
        if _tool_call_name(request) != self._tool_name:
            return
        args = _tool_call_args(request)
        cmd = args.get("command")
        reasons = screen_high_risk_command(cmd if isinstance(cmd, str) else None)
        if not reasons:
            return
        logger.warning(
            "HighRiskExecuteFlaggerMiddleware: high-risk command (reasons=%s) call_id=%s\n  cmd: %s",
            reasons, request.tool_call.get("id"), cmd,
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        self._log_if_risky(request)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        self._log_if_risky(request)
        return await handler(request)


# ── 4. Live output streaming ──────────────────────────────────────────────


class LiveOutputMiddleware(AgentMiddleware):
    """Stream ``execute`` command output lines to a caller-supplied queue.

    Wraps the ``execute`` command with ``tee`` so combined stdout/stderr
    flow to both the deepagents backend (unchanged semantics for the
    agent) **and** a per-session :class:`asyncio.Queue`.

    **Two-phase interrupt handling.** In deepagents/LangGraph, each tool
    call that has ``interrupt_on=True`` goes through the middleware
    chain *twice*:

    1. **Interrupt phase** — ``handler(request)`` raises a LangGraph
       ``NodeInterrupt``.  The HITL approval dialog is built from the
       tool-call args *at this point*, so we must **not** modify the
       command yet (the user should see the clean original command, not
       a ``tee``-wrapped one).

    2. **Execution phase** — after the user approves, the graph resumes
       and ``awrap_tool_call`` is called again with the same
       ``tool_call_id``.  Now it is safe to wrap with ``tee`` and start
       streaming.

    We distinguish the two phases by catching the interrupt exception on
    the first invocation and recording the ``tool_call_id`` in
    ``self._approved``.  On the second invocation we see the id in
    ``_approved`` and apply the ``tee`` wrapper transparently.
    """

    _TOOL_NAME = "execute"
    _POLL_INTERVAL = 0.1  # seconds between temp-file polls

    def __init__(self, queue: asyncio.Queue, files_dir: Path) -> None:
        self._queue = queue
        self._files_dir = files_dir
        # tool_call_ids that have already been through the HITL interrupt
        # and are now in the execution phase.
        self._approved: set[str] = set()

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        return handler(request)

    async def _poll_file(self, path: str, tc_id: str) -> None:
        """Poll *path* for new bytes and push decoded lines to the queue."""
        offset = 0
        line_idx = 0
        try:
            while True:
                await asyncio.sleep(self._POLL_INTERVAL)
                try:
                    with open(path, "rb") as fh:
                        fh.seek(offset)
                        data = fh.read()
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        for raw in text.splitlines(keepends=True):
                            await self._queue.put({
                                "type": "execute_output",
                                "content": raw.rstrip("\r\n"),
                                "metadata": {"tool_call_id": tc_id, "line_index": line_idx},
                            })
                            line_idx += 1
                        offset += len(data)
                except (OSError, FileNotFoundError):
                    break
        except asyncio.CancelledError:
            pass

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if _tool_call_name(request) != self._TOOL_NAME:
            return await handler(request)

        args = _tool_call_args(request)
        command = args.get("command")
        if not isinstance(command, str) or not command.strip():
            return await handler(request)

        tc_id: str = request.tool_call.get("id") or ""

        # ── Phase 1: interrupt phase ─────────────────────────────────────
        # Pass the request through unmodified so the HITL dialog shows
        # the original command.  Detect the interrupt exception and mark
        # this tc_id as approved so the next call (execution phase) knows
        # it's safe to apply the tee wrapper.
        if tc_id not in self._approved:
            try:
                return await handler(request)
            except BaseException as exc:
                # Any BaseException (NodeInterrupt, CancelledError, …)
                # that is NOT a normal return is treated as the interrupt
                # signal.  Mark this tc_id so the execution phase can
                # apply tee, then re-raise to let LangGraph handle it.
                if not isinstance(exc, Exception):
                    # True BaseException subclasses (CancelledError,
                    # SystemExit, KeyboardInterrupt) — always re-raise.
                    raise
                # For regular Exception subclasses (NodeInterrupt is
                # typically an Exception) mark as approved and re-raise.
                self._approved.add(tc_id)
                raise

        # ── Phase 2: execution phase (post-approval) ─────────────────────
        # The user approved the command.  Now wrap with tee and stream.
        self._approved.discard(tc_id)  # clean up for any future reuse

        tmp_path: Optional[str] = None
        poll_task: Optional[asyncio.Task] = None
        actual_request = request

        try:
            import shlex as _shlex
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self._files_dir,
                prefix=".live_",
                suffix=".log",
            )
            os.close(tmp_fd)
            safe_log = _shlex.quote(tmp_path)
            wrapped_cmd = f"{{ {command}; }} 2>&1 | tee {safe_log}"
            new_call = {**request.tool_call, "args": {**args, "command": wrapped_cmd}}
            actual_request = request.override(tool_call=new_call)
            poll_task = asyncio.create_task(self._poll_file(tmp_path, tc_id))
        except Exception:
            logger.debug(
                "LiveOutputMiddleware: streaming setup failed, running without",
                exc_info=True,
            )

        try:
            return await handler(actual_request)
        finally:
            if poll_task is not None and not poll_task.done():
                poll_task.cancel()
                await asyncio.gather(poll_task, return_exceptions=True)
            if tmp_path is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)


__all__ = [
    "ExecutePathSafetyMiddleware",
    "SubagentAsToolGuardMiddleware",
    "HighRiskExecuteFlaggerMiddleware",
    "LiveOutputMiddleware",
    "screen_high_risk_command",
]
