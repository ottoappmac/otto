"""Session lifecycle management — wires the agent with the selected agent config."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Iterator, Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from backend.config import AppConfig, get_app_data_dir
from backend.schemas import AgentSpec, SessionInfo
from backend.session_transcript import append_event_async as _transcript
from backend.utils import (
    extract_text_content,
    is_resolved_path_allowed,
    platform_label,
    remap_to_virtual_path,
)

logger = logging.getLogger(__name__)

# ---- Live output streaming helpers ---------------------------------------

_HEAD_LINES = 15  # lines shown from the start of execute output
_TAIL_LINES = 10  # lines shown from the end


def _head_tail_preview(text: str) -> tuple[str, int, bool]:
    """Return ``(preview, total_lines, truncated)`` for an execute result.

    When the output is short enough to fit in HEAD + TAIL lines the full
    text is returned unchanged.  Otherwise the middle is replaced with a
    compact ``⋯ (N more lines) ⋯`` separator so the agent and the UI both
    see the most informative parts of long output without the full blob.
    """
    lines = text.splitlines(keepends=True)
    total = len(lines)
    if total <= _HEAD_LINES + _TAIL_LINES:
        return text, total, False
    omitted = total - _HEAD_LINES - _TAIL_LINES
    preview = (
        "".join(lines[:_HEAD_LINES])
        + f"\n\u22ef ({omitted} more lines) \u22ef\n"
        + "".join(lines[-_TAIL_LINES:])
    )
    return preview, total, True


# ---- Shell backend with SESSION_FILES env var ----------------------------


def _make_backend(files_dir: Path) -> Any:
    """Create a ``LocalShellBackend`` with ``SESSION_FILES`` injected into
    the shell environment so agents can reference the real session files path.

    The ``execute`` tool's working directory is already set to *files_dir*,
    so relative paths work too.  ``SESSION_FILES`` provides an explicit
    anchor for cases where the agent needs a full absolute path.

    We subclass to relax the symlink-resolution check so that user-dropped
    files/folders, which we expose as symlinks under ``files_dir/links/``,
    can be reached via the standard virtual-path tools (ls, read_file,
    grep, glob, view_image).  The upstream backend calls ``Path.resolve()``
    before the containment check, which follows symlinks and causes any
    symlink pointing outside ``files_dir`` to be rejected.  Our override
    validates containment *lexically* — ``..`` is still blocked, so the
    only thing that changes is "symlinks may legitimately escape".
    """
    from deepagents.backends.local_shell import LocalShellBackend

    class _SymlinkAwareBackend(LocalShellBackend):
        def _resolve_path(self, key: str) -> Path:
            if not self.virtual_mode:
                return super()._resolve_path(key)
            # The agent is handed the REAL files_dir via the system prompt and
            # the SESSION_FILES env var, so it naturally builds absolute paths
            # like "{files_dir}/output/report.md".  Remap any absolute path that
            # genuinely lives inside the session root back to its virtual form
            # ("/output/report.md") instead of rejecting it below.  This only
            # normalizes the spelling of in-root paths; anything outside the
            # root is left unchanged and still hits the real-looking guard.
            key = remap_to_virtual_path(key, self.cwd)
            vpath = key if key.startswith("/") else "/" + key
            if ".." in vpath or vpath.startswith("~"):
                raise ValueError("Path traversal not allowed")
            # Reject paths that look like real absolute paths escaping the
            # virtual root (e.g. an agent calling write_file with a literal
            # filesystem path such as "/etc/passwd" that is NOT under the
            # session root).  Such paths would be silently mangled by the
            # lexical join below — stripped of their leading "/" and written
            # inside the session directory instead of the intended real
            # location — with no error surfaced to the agent.  Catching this
            # early gives a clear, actionable signal.
            if not key.startswith("/") or not any(
                vpath.lstrip("/").startswith(p)
                for p in ("output", "input", "links", "files")
            ):
                # Allow /output/, /input/, /links/, /files/ virtual mounts
                # and bare relative paths.  Block anything that starts with
                # a real-looking absolute path component (e.g. /Users, /home,
                # /var, /tmp, /private) that isn't under those mounts.
                real_looking = any(
                    vpath.startswith(f"/{seg}/") or vpath == f"/{seg}"
                    for seg in ("Users", "home", "var", "tmp", "private", "opt", "usr", "etc")
                )
                if real_looking:
                    raise ValueError(
                        f"Path '{key}' is outside the session files directory and "
                        f"cannot be used with file tools.  Save output under "
                        f"'/output/<name>' (or use a path relative to the session "
                        f"root); the file tools operate inside the virtual session "
                        f"directory, not the host filesystem."
                    )
            # Lexical join — DO NOT call .resolve() here, otherwise symlinks
            # that legitimately point outside the session (user-dropped
            # context under /links/) would be rejected by the containment
            # check below.
            full = self.cwd / vpath.lstrip("/")
            try:
                full.relative_to(self.cwd)
            except ValueError:
                raise ValueError(
                    f"Path:{full} outside root directory: {self.cwd}"
                ) from None
            # Defense-in-depth: after the lexical check, re-resolve symlinks so
            # a link planted inside the session root cannot be used to escape
            # it.  User-dropped escapes are legitimate only under /links/.
            if not is_resolved_path_allowed(full, self.cwd, vpath):
                raise ValueError(
                    f"Path '{key}' resolves outside the session directory via a "
                    f"symlink and cannot be used with file tools."
                )
            # Return the UNRESOLVED path so symlink-backed /links/ paths keep
            # their virtual spelling in _to_virtual_path.
            return full

        # The parent class methods (glob, ls, read, write, edit) call
        # _resolve_path outside any try/except, so a ValueError raised for
        # real-looking paths (e.g. /Users/…) propagates uncaught through
        # FilesystemMiddleware all the way to the session stream and crashes
        # the entire run.  Override each method to catch ValueError from path
        # resolution and return a safe error/empty result instead.

        def glob(self, pattern: str, path: str = "/") -> Any:
            try:
                return super().glob(pattern, path)
            except ValueError as exc:
                from deepagents.backends.protocol import GlobResult
                logger.warning("[session_backend] glob blocked: %s", exc)
                return GlobResult(matches=[])

        def ls(self, path: str) -> Any:
            try:
                return super().ls(path)
            except ValueError as exc:
                from deepagents.backends.protocol import LsResult
                logger.warning("[session_backend] ls blocked: %s", exc)
                return LsResult(entries=[])

        def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:
            try:
                return super().read(file_path, offset, limit)
            except ValueError as exc:
                from deepagents.backends.protocol import ReadResult
                logger.warning("[session_backend] read blocked: %s", exc)
                return ReadResult(error=str(exc))

        def write(self, file_path: str, content: str) -> Any:
            try:
                return super().write(file_path, content)
            except ValueError as exc:
                from deepagents.backends.protocol import WriteResult
                logger.warning("[session_backend] write blocked: %s", exc)
                return WriteResult(error=str(exc))

        def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> Any:
            try:
                return super().edit(file_path, old_string, new_string, replace_all)
            except ValueError as exc:
                from deepagents.backends.protocol import EditResult
                logger.warning("[session_backend] edit blocked: %s", exc)
                return EditResult(error=str(exc))

        def _to_virtual_path(self, path: Path) -> str:
            # Lexical conversion: when iterating a symlinked dir, we want
            # ``/links/foo/bar.txt`` not ``/Users/.../bar.txt``.  The
            # caller passes us a child path that's already inside cwd
            # lexically (we joined cwd + key without resolving), so we
            # just strip the cwd prefix without follow-symlink resolution.
            try:
                rel = path.relative_to(self.cwd)
            except ValueError:
                # Fallback to upstream behaviour for paths that genuinely
                # need .resolve() (e.g. recursive shell-glob results).
                rel = path.resolve().relative_to(self.cwd)
            return "/" + rel.as_posix()

    return _SymlinkAwareBackend(
        root_dir=files_dir,
        virtual_mode=True,
        inherit_env=True,
        env={"SESSION_FILES": str(files_dir)},
    )


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _validate_session_id(session_id: str) -> None:
    if not _UUID_RE.match(session_id):
        raise ValueError(f"Invalid session ID format: {session_id!r}")


def _sessions_dir() -> Path:
    d = get_app_data_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session_files_dir(session_id: str) -> Path:
    _validate_session_id(session_id)
    d = _sessions_dir() / session_id / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _checkpoint_db_path() -> str:
    return str(_sessions_dir() / "checkpoints.sqlite")


async def _connect_checkpoint_db() -> aiosqlite.Connection:
    """Open the checkpoint DB with incremental auto-vacuum enabled.

    Without this, ``DELETE``s issued by ``delete_session`` free pages only
    into SQLite's internal freelist — the file itself never shrinks, so a
    session-heavy install accumulates an ever-growing ``checkpoints.sqlite``
    even though the live row count stays small. ``auto_vacuum=INCREMENTAL``
    only takes effect after a ``VACUUM``, so this is a no-op on a database
    that already has it set (the common case); it only matters the first
    time this runs against a legacy NONE-mode file, or against a brand new
    one.
    """
    conn = await aiosqlite.connect(_checkpoint_db_path())
    await conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
    return conn


def _messages_path(session_id: str) -> Path:
    _validate_session_id(session_id)
    return _sessions_dir() / f"{session_id}.messages.json"


def load_messages(session_id: str) -> list[dict[str, Any]]:
    p = _messages_path(session_id)
    if not p.exists():
        return []
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _save_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    p = _messages_path(session_id)
    with p.open("w", encoding="utf-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")


def _append_message(session_id: str, msg: dict[str, Any]) -> None:
    p = _messages_path(session_id)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, default=str) + "\n")


async def _append_message_async(session_id: str, msg: dict[str, Any]) -> None:
    """Non-blocking wrapper — moves the sync file write to a thread."""
    await asyncio.to_thread(_append_message, session_id, msg)


async def _load_messages_async(session_id: str) -> list[dict[str, Any]]:
    """Non-blocking wrapper for load_messages."""
    return await asyncio.to_thread(load_messages, session_id)


async def _save_messages_async(session_id: str, messages: list[dict[str, Any]]) -> None:
    """Non-blocking wrapper for _save_messages."""
    await asyncio.to_thread(_save_messages, session_id, messages)


_PLAYWRIGHT_SERVER_ID = "playwright-mcp"


async def _maybe_create_pw_pool(mcp_mgr: Any) -> Any:
    """Create a :class:`PlaywrightPool` and proxy the primary PW tools.

    Returns the pool (also stored on *mcp_mgr* for cleanup), or ``None``
    when Playwright MCP is not connected.
    """
    connections = mcp_mgr.connections
    pw_conn = connections.get(_PLAYWRIGHT_SERVER_ID)
    if pw_conn is None or not pw_conn.connected:
        return None

    from backend.playwright_pool import PlaywrightPool, proxy_playwright_tools

    pool = PlaywrightPool(pw_conn.config)
    await pool.startup()
    mcp_mgr.pw_pool = pool
    proxy_playwright_tools(pw_conn.tools)
    logger.info("Playwright pool created (max %d instances), tools proxied", pool._max_instances)
    return pool


def _make_prompt_caching_middleware(model: Any) -> Any:
    """Return the correct prompt-caching middleware for *model*.

    - ``ChatBedrockConverse`` → ``BedrockPromptCachingMiddleware``
    - ``ChatAnthropic``       → ``None`` (deepagents already includes
                                ``AnthropicPromptCachingMiddleware`` in its
                                standard stack; adding it again triggers
                                "Please remove duplicate middleware instances")
    - Anything else           → ``None`` (caller should omit from stack)
    """
    from langchain_aws import ChatBedrockConverse

    if isinstance(model, ChatBedrockConverse):
        from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
        return BedrockPromptCachingMiddleware(ttl="5m", unsupported_model_behavior="ignore")

    logger.debug("No prompt-caching middleware for model type %s", type(model).__name__)
    return None


def _maybe_tool_image_relocation(model: Any) -> Any | None:
    """Return a ``ToolImageRelocationMiddleware`` for OpenAI-compatible models.

    See :mod:`middleware.tool_image_relocation` for why tool-result images
    must be relocated into user messages for these providers.
    """
    from middleware.tool_image_relocation import maybe_for_model

    return maybe_for_model(model)


def _maybe_file_attachment_filename(model: Any) -> Any | None:
    """Return a ``FileAttachmentFilenameMiddleware`` for OpenAI-compatible models.

    See :mod:`middleware.file_attachment_filename` for why ``read_file``
    attachments (PDF, PPTX, etc.) need a real filename filled in for these
    providers — without it, some servers (e.g. oMLX) reject the request as
    an "Unsupported attachment type".
    """
    from middleware.file_attachment_filename import maybe_for_model

    return maybe_for_model(model)


def _maybe_react_shim(model: Any) -> Any | None:
    """Return a ReAct-shim middleware instance when *model* lacks native tool calling.

    Text-only on-device models (``ChatMLXText`` with a
    non-tool-aware chat template, …) inherit ``BaseChatModel.bind_tools``
    which raises ``NotImplementedError``.  The langchain agent factory calls
    ``model.bind_tools(...)`` before every model invocation, so without a
    shim the very first turn crashes.

    ``MLXReActMiddleware`` is misnamed but generic: it sets ``tools=[]`` so
    the framework skips ``bind_tools`` entirely, injects a ReAct tool-use
    section into the system prompt, and parses ``Action: ```json{...}`````
    blocks back into structured ``tool_calls``.  Its
    :meth:`_needs_react_shim` self-check returns ``True`` for text-only
    models (some MLX templates), ``False`` for models with a real
    ``bind_tools`` override (Anthropic, OpenAI, Bedrock, EXO/oMLX via
    ChatOpenAI, …) — so we can call it unconditionally and let it
    self-gate.

    Returned at index 0 of the middleware list so it wraps every other
    middleware and intercepts the model request before the deepagents stack
    tries to bind tools.
    """
    try:
        from middleware.react_middleware import MLXReActMiddleware
    except ImportError:
        return None
    if not MLXReActMiddleware._needs_react_shim(model):
        return None

    # Small context-window models cannot fit the full argument schemas for
    # 50+ tools.  Switch to compact one-line-per-tool rendering and cap the
    # tool count so the system prompt (template + tool list) stays well
    # under the model's input budget.
    # Heuristic: if max_input_tokens < 8 000, use compact mode.
    context_window = _model_input_budget(model)

    if context_window is not None and context_window < 8000:
        # Reserve ~300 tokens for the ReAct template boilerplate; each compact
        # tool line is ~15–25 tokens, so floor(budget / 20) tools fit safely.
        budget = max(0, context_window - 300)
        max_tools = max(1, budget // 20)
        return MLXReActMiddleware(compact_tools=True, max_tools=max_tools)

    return MLXReActMiddleware()


def _model_input_budget(model: Any) -> int | None:
    """Return ``model.profile["max_input_tokens"]`` if exposed, else ``None``."""
    profile = getattr(model, "profile", None)
    if isinstance(profile, dict):
        value = profile.get("max_input_tokens")
        if isinstance(value, int) and value > 0:
            return value
    return None


def _maybe_context_truncation(model: Any) -> Any | None:
    """Return a ``SmallContextTruncationMiddleware`` instance when *model*
    has a sub-8 K input budget.

    The deepagents framework hardcodes :class:`TodoListMiddleware`,
    :class:`FilesystemMiddleware`, and :class:`SubAgentMiddleware`, each
    of which injects its own multi-hundred-token system-prompt block.
    On small on-device models (e.g. tiny MLX templates) those blocks
    plus Otto's lite orchestrator prompt and the ReAct tool section can
    exceed the model's capacity before any user message is added.

    This middleware is appended at the END of the extra-middleware list
    so it runs innermost (closest to the model) and observes the fully
    assembled request after every other middleware has run.  It clips
    the system message tail and drops oldest conversation messages until
    the request fits.
    """
    try:
        from middleware.context_truncation import SmallContextTruncationMiddleware
    except ImportError:
        return None
    budget = _model_input_budget(model)
    if budget is None or budget >= 8000:
        return None
    return SmallContextTruncationMiddleware(max_input_tokens=budget)


def _build_standard_tools(
    llm: Any = None,
    *,
    files_dir: Path | None = None,
    backend: Any = None,
    extract_images: bool = False,
) -> list[Any]:
    """Load standard research tools available to all agents (main and subagents).

    When *extract_images* is True (a vision model is available for the
    session), the document tools extract embedded images from PDF/DOCX/PPTX
    files into ``{files_dir}/doc_images`` and reference them inline so the
    agent can inspect them with ``view_image``.
    """
    tools: list[Any] = []
    try:
        from tools.large_results_grep import create_grep_large_results_tool
        tools.append(create_grep_large_results_tool(backend))
    except Exception as exc:
        logger.warning("Standard tool grep_large_results: could not load — %s", exc)
    try:
        from tools.research.web_researcher import web_research
        tools.append(web_research)
    except Exception as exc:
        logger.warning("Standard tool web_researcher: could not load — %s", exc)
    try:
        from tools.research.youtube_transcript import (
            youtube_search,
            youtube_transcript,
        )
        tools.append(youtube_search)
        tools.append(youtube_transcript)
    except Exception as exc:
        logger.warning("Standard tool youtube_transcript: could not load — %s", exc)
    try:
        from tools.research.doc_researcher import build_doc_research
        tools.append(build_doc_research(
            files_dir=files_dir,
            extract_images=extract_images,
        ))
    except Exception as exc:
        logger.warning("Standard tool doc_researcher: could not load — %s", exc)
    try:
        from tools.research.semantic_search import semantic_search
        tools.append(semantic_search)
    except Exception as exc:
        logger.warning("Standard tool semantic_search: could not load — %s", exc)
    if llm is not None:
        try:
            from tools.research.doc_reader import DocReader
            tools.append(DocReader.from_llm(
                llm,
                files_dir=files_dir,
                extract_images=extract_images,
            ))
        except Exception as exc:
            logger.warning("Standard tool doc_reader: could not load — %s", exc)
    return tools


def _apply_universal_loop_guard(
    tools: list[Any], *, scope: str, session_id: str | None = None
) -> Any:
    """Universal loop-guard chokepoint for an agent's final tool list.

    Wraps every tool with one shared :class:`ToolLoopGuard` so that no tool
    can reach the model unguarded.  Tools that a per-connection loader (MCP,
    macOS-native, Playwright) already wrapped are skipped idempotently and
    keep their specialised guard; only the directly-loaded tools (research,
    file, management, etc.) get newly guarded here.  Returns the guard (or
    ``None`` on failure) so callers can hold a reference for escalation.

    When *session_id* is given, an escalation callback is installed so that a
    guard which keeps tripping past ``max_escalations`` marks the session for
    a cooperative abort (see :func:`backend.streaming_subagent` step boundary).
    """
    try:
        from tools.loop_guard import guard_all_tools
        from utilities.environment import Environment

        on_escalate = None
        if session_id is not None:
            from backend.streaming_subagent import request_loop_abort

            def on_escalate(reason: str) -> None:
                request_loop_abort(session_id, reason)

        return guard_all_tools(
            tools,
            recovery_hint=(
                "You are repeating tool calls without making progress. Stop "
                "and either try a fundamentally different approach/tool or "
                "report your best answer from what you already have."
            ),
            window=Environment.get_loop_guard_window(),
            max_no_progress=Environment.get_loop_guard_max_no_progress(),
            max_identical_success=Environment.get_loop_guard_max_success(),
            recovery_temperature=Environment.get_loop_recovery_temperature(),
            recovery_temperature_turns=(
                Environment.get_loop_recovery_temperature_turns()
            ),
            max_escalations=Environment.get_loop_guard_max_escalations(),
            on_escalate=on_escalate,
        )
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "Universal loop guard (%s): could not apply — %s", scope, exc
        )
        return None


def _insert_repeated_thought_guard(middleware: list[Any], *, scope: str) -> None:
    """Insert the repeated-thought guard just inside any leading ReAct shim.

    The :class:`RepeatedThoughtGuardMiddleware` complements the per-tool
    :class:`tools.loop_guard.ToolLoopGuard` by catching loops whose tool calls
    keep *succeeding* with a jittering result (so neither the failure-loop nor
    the no-progress detector trips) but whose *thought + action* is identical
    every turn.  It nudges + bumps temperature, then ends the run with a
    partial result — which also unblocks an orchestrator waiting on a stuck
    subagent in a parallel ``task`` batch.

    Placed immediately after the ReAct shim (when present) so it counts repeats
    against the rawest available history and its abort short-circuit prevents
    the model call before inner middleware (summarization, truncation) run.
    No-op-safe: any import/wiring failure is logged and skipped.
    """
    try:
        from middleware.repeated_thought_guard import (
            RepeatedThoughtGuardMiddleware,
        )
        from middleware.react_middleware import MLXReActMiddleware

        idx = 0
        for i, mw in enumerate(middleware):
            if isinstance(mw, MLXReActMiddleware):
                idx = i + 1
                break
        middleware.insert(idx, RepeatedThoughtGuardMiddleware())
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "Repeated-thought guard (%s): could not apply — %s", scope, exc
        )

    # The intra-message repetition guard is provider-agnostic and sanitises the
    # model's *output*; inserting it next to the thought guard covers the
    # orchestrator and every subagent in one place.
    try:
        from middleware.repetition_guard import RepetitionGuardMiddleware
        from middleware.react_middleware import MLXReActMiddleware

        idx = 0
        for i, mw in enumerate(middleware):
            if isinstance(mw, MLXReActMiddleware):
                idx = i + 1
                break
        middleware.insert(idx, RepetitionGuardMiddleware())
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "Repetition guard (%s): could not apply — %s", scope, exc
        )

    # The per-run tool-call budget nudges then gracefully stops a run that
    # thrashes through hundreds of (non-identical) tool calls — a pattern the
    # per-tool loop guard misses.  Scoped per graph invocation via the request
    # history, so it covers the orchestrator and every subagent.
    try:
        from middleware.tool_call_budget import ToolCallBudgetMiddleware
        from middleware.react_middleware import MLXReActMiddleware

        idx = 0
        for i, mw in enumerate(middleware):
            if isinstance(mw, MLXReActMiddleware):
                idx = i + 1
                break
        middleware.insert(idx, ToolCallBudgetMiddleware())
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "Tool-call budget guard (%s): could not apply — %s", scope, exc
        )


def _build_gp_subagent(
    model: Any,
    tools: list[Any],
    backend: Any,
    *,
    pw_pool: Any = None,
    memory_middleware: list[Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build the general-purpose subagent as a ``CompiledSubAgent``.

    By compiling it here (instead of letting ``create_deep_agent`` do it
    internally) we gain two things:

    1. The graph is wrapped in :class:`StreamingSubagentRunnable` so tool
       calls / results stream to the UI in real time.
    2. ``HumanInTheLoopMiddleware`` is intentionally omitted — subagents
       run inside the ``task`` tool where HITL interrupts have no channel
       back to the user, causing the subagent to hang forever.

    ``create_deep_agent`` skips its built-in GP subagent when it sees one
    with the same name already present in the ``subagents`` list.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        compute_summarization_defaults,
    )
    from deepagents.middleware.subagents import (
        DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
        DEFAULT_SUBAGENT_PROMPT,
    )
    from backend.streaming_subagent import StreamingSubagentRunnable

    summ_defaults = compute_summarization_defaults(model)
    middleware: list[Any] = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=summ_defaults["trigger"],
            keep=summ_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summ_defaults["truncate_args_settings"],
        ),
    ]
    if memory_middleware:
        middleware.extend(memory_middleware)
    caching_mw = _make_prompt_caching_middleware(model)
    if caching_mw:
        middleware.append(caching_mw)
    middleware.append(PatchToolCallsMiddleware())

    # Runtime safety guards: rewrite bare /output paths in execute(),
    # log high-risk commands, and (orchestrator-only) repair
    # subagent-as-tool calls.  These promote the highest-risk
    # prompt-only rules into runtime invariants so the slim lite prompt
    # is safe.
    from backend.safety_middleware import (
        ExecutePathSafetyMiddleware,
        HighRiskExecuteFlaggerMiddleware,
    )
    middleware.append(ExecutePathSafetyMiddleware())
    middleware.append(HighRiskExecuteFlaggerMiddleware())

    # When the general-purpose subagent owns the Playwright tools (i.e. no
    # library sub-agent claimed ``playwright-mcp``), attach the snapshot
    # pruning middleware so large accessibility-tree YAML payloads don't
    # balloon the context window.
    if pw_pool is not None:
        from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware
        middleware.insert(0, PlaywrightSnapshotPruningMiddleware())
        logger.info("Playwright snapshot pruning enabled for general-purpose subagent")

    # ReAct shim for text-only on-device models — same rationale as the
    # main orchestrator path (see ``_maybe_react_shim``).  Insert at 0 so
    # it wraps the rest of the stack.
    react_shim = _maybe_react_shim(model)
    if react_shim is not None:
        middleware.insert(0, react_shim)
        logger.info(
            "ReAct shim enabled for general-purpose subagent (model=%s)",
            type(model).__name__,
        )

    # Relocate view_image tool-result images into a following user message for
    # OpenAI-compatible models — the GP subagent owns the file tools in
    # general-purpose mode, so this is where view_image usually runs.  No-op
    # for providers that render tool-result images natively.
    image_relocation = _maybe_tool_image_relocation(model)
    if image_relocation is not None:
        middleware.append(image_relocation)
        logger.info(
            "Tool-image relocation enabled for general-purpose subagent (model=%s)",
            type(model).__name__,
        )

    # Fill in real filenames on read_file's PDF/PPTX attachments so
    # OpenAI-compatible servers don't reject them as an unsupported type.
    file_attachment_filename = _maybe_file_attachment_filename(model)
    if file_attachment_filename is not None:
        middleware.append(file_attachment_filename)

    # Last-resort context-window enforcement.  Appended at the END so it
    # runs innermost — sees the request after every other middleware
    # (including deepagents' hardcoded Todo/Filesystem/SubAgents stack)
    # has injected its system-prompt block.  No-op for big-context models.
    ctx_trunc = _maybe_context_truncation(model)
    if ctx_trunc is not None:
        middleware.append(ctx_trunc)
        logger.info(
            "Context truncation enabled for general-purpose subagent (model=%s, budget=%d tok)",
            type(model).__name__,
            ctx_trunc._budget,
        )

    # Pick a slim prompt for the GP subagent when it runs on an
    # open-source local provider — same rationale as the orchestrator
    # lite path.  The GP subagent only needs to know it should use the
    # provided tools and return a concise answer; the long DEFAULT
    # prompt's discipline rules are mostly redundant with the guards
    # already attached to this stack (path safety, summarisation,
    # high-risk logging).
    from utilities.environment import Environment
    _CITATION_GUIDANCE = (
        "\n\nWhen your response draws on web research, embed the source as an inline "
        "Markdown link [text](url) directly on the relevant text — companies, standards, "
        "regulations, prices, statistics, or any verifiable claim. Never fabricate URLs; "
        "only link pages you actually retrieved during this task."
    )
    if Environment.use_lite_orchestrator_prompt():
        gp_prompt = (
            "You are the general-purpose subagent. Use the provided tools to "
            "complete the task. Save large outputs to /output/ via write_file "
            "and return a concise final answer."
            + _CITATION_GUIDANCE
        )
    else:
        gp_prompt = DEFAULT_SUBAGENT_PROMPT + _CITATION_GUIDANCE

    _apply_universal_loop_guard(
        tools, scope="general-purpose", session_id=session_id
    )
    _insert_repeated_thought_guard(middleware, scope="general-purpose")
    graph = create_agent(
        model,
        system_prompt=gp_prompt,
        tools=tools,
        middleware=middleware,
        name="general-purpose",
    )

    return {
        "name": "general-purpose",
        "description": DEFAULT_GENERAL_PURPOSE_DESCRIPTION,
        "runnable": StreamingSubagentRunnable(graph, "general-purpose", pw_pool=pw_pool),
    }


_RESERVED_SUBAGENT_NAMES = frozenset({"general-purpose"})


def _build_named_agent_subagent(
    agent_name: str,
    agent_spec: Any,
    model: Any,
    tools: list[Any],
    backend: Any,
    *,
    pw_pool: Any = None,
    memory_middleware: list[Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any] | None:
    """Build a ``CompiledSubAgent`` that runs *agent_name* as a subagent.

    Used in direct-agent mode so the orchestrator can call
    ``task(subagent_type="<agent_name>")`` to delegate subtasks to the
    same agent it is configured to run as.  Without this, only
    ``general-purpose`` would be available via the ``task`` tool when an
    agent is selected directly (e.g. via a schedule's ``agent_name``).

    Returns ``None`` when the agent spec cannot be loaded or has no
    description (no point advertising it if we can't describe it).
    """
    if agent_spec is None:
        return None

    description = getattr(agent_spec, "description", None) or ""
    if not description:
        return None

    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        compute_summarization_defaults,
    )
    from backend.agent_library import get_agent_system_prompt
    from backend.streaming_subagent import StreamingSubagentRunnable

    system_prompt = get_agent_system_prompt(agent_name) or ""

    summ_defaults = compute_summarization_defaults(model)
    middleware: list[Any] = [
        TodoListMiddleware(),
        FilesystemMiddleware(backend=backend),
        SummarizationMiddleware(
            model=model,
            backend=backend,
            trigger=summ_defaults["trigger"],
            keep=summ_defaults["keep"],
            trim_tokens_to_summarize=None,
            truncate_args_settings=summ_defaults["truncate_args_settings"],
        ),
    ]
    if memory_middleware:
        middleware.extend(memory_middleware)
    caching_mw = _make_prompt_caching_middleware(model)
    if caching_mw:
        middleware.append(caching_mw)
    middleware.append(PatchToolCallsMiddleware())

    from backend.safety_middleware import (
        ExecutePathSafetyMiddleware,
        HighRiskExecuteFlaggerMiddleware,
    )
    middleware.append(ExecutePathSafetyMiddleware())
    middleware.append(HighRiskExecuteFlaggerMiddleware())

    if pw_pool is not None:
        from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware
        middleware.insert(0, PlaywrightSnapshotPruningMiddleware())

    react_shim = _maybe_react_shim(model)
    if react_shim is not None:
        middleware.insert(0, react_shim)

    image_relocation = _maybe_tool_image_relocation(model)
    if image_relocation is not None:
        middleware.append(image_relocation)

    file_attachment_filename = _maybe_file_attachment_filename(model)
    if file_attachment_filename is not None:
        middleware.append(file_attachment_filename)

    ctx_trunc = _maybe_context_truncation(model)
    if ctx_trunc is not None:
        middleware.append(ctx_trunc)

    _apply_universal_loop_guard(tools, scope=agent_name, session_id=session_id)
    _insert_repeated_thought_guard(middleware, scope=agent_name)
    graph = create_agent(
        model,
        system_prompt=system_prompt,
        tools=tools,
        middleware=middleware,
        name=agent_name,
    )

    return {
        "name": agent_name,
        "description": description,
        "runnable": StreamingSubagentRunnable(graph, agent_name, pw_pool=pw_pool),
    }


def _reclaim_mlx_memory() -> None:
    """Best-effort GC + Metal allocator reclaim.

    Runs in a worker thread (it's CPU-bound and ``mx.clear_cache`` can block
    briefly).  No-op / silently ignored on non-Apple-Silicon hosts where
    ``mlx.core`` isn't importable.  ``mx.clear_cache`` only frees the unused
    portion of the allocator pool, so calling it while MLX is still the
    active provider does not evict in-use weights.
    """
    import gc

    gc.collect()
    try:
        import mlx.core as mx  # type: ignore[import-untyped]

        # Serialise the allocator reclaim against any in-flight generation or
        # model warmup.  ``mx.clear_cache`` mutating the Metal allocator pool
        # while another thread is mid-``stream_generate`` (e.g. a fresh model
        # warmup) raced into a "Command buffer execution failed: Insufficient
        # Memory" abort that killed the whole process.  ``MLX_GEN_LOCK`` is the
        # same process-wide lock that every ``stream_generate`` / warmup call
        # holds, so taking it here guarantees the two never overlap.
        from chat_models.mlx._shared import MLX_GEN_LOCK  # noqa: PLC0415

        with MLX_GEN_LOCK:
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            else:  # older mlx releases
                mx.metal.clear_cache()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


@contextlib.contextmanager
def _hf_llm_id_override(repo_id: str | None) -> Iterator[None]:
    """Temporarily set ``HF_LLM_MODEL_ID`` for a single ``create_llm("mlx")`` call."""
    rid = (repo_id or "").strip()
    if not rid:
        yield
        return
    key = "HF_LLM_MODEL_ID"
    prev = os.environ.get(key)
    os.environ[key] = rid
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


def _build_orch_llm_sync(config: Any, base_llm: Any) -> Any:
    """Resolve the orchestrator model from ``config.orchestrator``.

    When ``orchestrator.llm_family`` is ``"follow_main"`` (the default) the
    main chat LLM (``base_llm``) is returned unchanged.  For any other value
    the matching provider is instantiated.  This is the model that:

    * The main agent graph uses as its reasoning engine.
    * ``"inherit"`` (and ``None``) subagents receive as their ``parent_model``
      so they truly share the orchestrator's provider rather than always
      inheriting the main chat provider.

    If the orchestrator provider cannot be instantiated (e.g. the MLX model
    weights have not been downloaded yet) we log a warning and fall back to
    ``base_llm`` so that session creation never hard-fails due to an
    out-of-date orchestrator setting.

    Must be called inside ``asyncio.to_thread`` when the orchestrator family
    is ``"mlx"`` (weight loading blocks the event loop).
    """
    from deep_agent.model_factory import create_llm as _create_llm
    from utilities.environment import Environment

    orch = config.orchestrator

    try:
        # Legacy provider_override wins over the newer llm_family field.
        po = (orch.provider_override or "").strip().lower()
        if po:
            return _create_llm(po)

        fam = (orch.llm_family or "follow_main").strip().lower()
        if fam in ("follow_main", ""):
            return base_llm
        if fam == "frontier":
            return _create_llm("anthropic")
        if fam == "openai":
            return _create_llm("openai")
        if fam == "exo":
            return _create_llm("exo")
        if fam == "mlx":
            # Skip a second in-process MLX load while the main chat provider
            # is a cloud/frontier one — loading another full set of weights
            # (plus warmup) into unified Metal memory is what caused GPU OOM
            # aborts after switching away from MLX.  Ride on the main model.
            if not Environment.is_oss_local_provider(config.llm.provider):
                logger.info(
                    "_build_orch_llm_sync: orchestrator llm_family='mlx' but main "
                    "provider=%r is non-local — reusing main chat model to avoid a "
                    "second MLX load.",
                    config.llm.provider,
                )
                return base_llm
            with _hf_llm_id_override(orch.mlx_model or ""):
                return _create_llm("mlx")
        # Unknown family — fall back to the main chat model.
        logger.warning(
            "_build_orch_llm_sync: unknown orchestrator llm_family=%r, using main chat model",
            fam,
        )
        return base_llm
    except Exception as exc:
        logger.warning(
            "_build_orch_llm_sync: could not create orchestrator model "
            "(llm_family=%r) — falling back to main chat model. Reason: %s",
            getattr(orch, "llm_family", None),
            exc,
        )
        return base_llm


def _resolve_subagent_model(agent_spec: AgentSpec, parent_model: Any) -> Any:
    """Pick the chat model for a library-backed ``task`` subagent."""
    from deep_agent.model_factory import create_llm
    from deepagents._models import resolve_model
    from utilities.environment import Environment

    fam_raw = agent_spec.subagent_llm_family
    fam = fam_raw.strip().lower() if isinstance(fam_raw, str) and fam_raw.strip() else None

    if fam is None:
        if agent_spec.model_override:
            return resolve_model(agent_spec.model_override)
        return parent_model

    if fam == "inherit":
        return parent_model
    if fam == "frontier":
        return create_llm("anthropic")
    if fam == "mlx":
        # Skip a second in-process MLX load while the main provider is cloud —
        # see _build_orch_llm_sync for the GPU-OOM rationale.
        if not Environment.is_oss_local_provider():
            logger.info(
                "Agent %r has subagent_llm_family='mlx' but main provider is "
                "non-local — reusing parent model to avoid a second MLX load.",
                agent_spec.name,
            )
            return parent_model
        mid = (agent_spec.mlx_model_id or "").strip()
        if mid:
            with _hf_llm_id_override(mid):
                return create_llm("mlx")
        return create_llm("mlx")
    if fam == "exo":
        # exo is a single shared cluster — every subagent invocation hits
        # the same OpenAI-compatible endpoint and the same model id
        # (Settings → LLM → Models). No per-subagent model override
        # because the cluster only keeps one instance loaded per id.
        return create_llm("exo")
    if fam == "custom":
        if not agent_spec.model_override:
            logger.warning(
                "Agent %r has subagent_llm_family=custom but no model_override; using main model",
                agent_spec.name,
            )
            return parent_model
        return resolve_model(agent_spec.model_override)

    logger.warning(
        "Unknown subagent_llm_family=%r for agent %r; using main model",
        fam_raw,
        agent_spec.name,
    )
    return parent_model


def _split_provider_model(model_override: str) -> tuple[str, str]:
    """Best-effort split of a ``provider:model`` override into its parts.

    ``resolve_model`` accepts either ``provider:model`` or a bare model id
    (``init_chat_model`` infers the provider).  For vision detection we only
    need a best-effort provider; an empty provider makes
    :func:`supports_vision` return ``False`` (conservative).
    """
    s = (model_override or "").strip()
    if ":" in s:
        prov, _, mid = s.partition(":")
        return prov.strip().lower(), mid.strip()
    return "", s


def _provider_model_for_main(config: Any) -> tuple[str, str]:
    """Return ``(provider, model_id)`` for the active main chat model."""
    p = config.llm.provider
    if p == "omlx":
        return p, getattr(config.omlx, "model_name", "") or ""
    if p == "exo":
        return p, getattr(config.exo, "model_name", "") or ""
    if p == "mlx":
        return p, getattr(config.llm.mlx, "hf_llm_model_id", "") or ""
    if p == "anthropic":
        return p, getattr(config.llm.anthropic, "model_name", "") or ""
    if p == "openai":
        return p, getattr(config.llm.openai, "model_name", "") or ""
    return p, ""


def _provider_model_for_orchestrator(config: Any) -> tuple[str, str]:
    """Return ``(provider, model_id)`` for the orchestrator model.

    The orchestrator is the ``parent_model`` that ``inherit``/``None`` subagents
    receive, so its capability is what those subagents inherit.  Mirrors the
    family branches in :func:`_build_orch_llm_sync`.
    """
    from utilities.environment import Environment

    orch = config.orchestrator
    po = (getattr(orch, "provider_override", "") or "").strip().lower()
    fam = po or (getattr(orch, "llm_family", "") or "follow_main").strip().lower()
    if fam in ("follow_main", ""):
        return _provider_model_for_main(config)
    if fam in ("frontier", "anthropic"):
        return "anthropic", getattr(config.llm.anthropic, "model_name", "") or ""
    if fam == "openai":
        return "openai", getattr(config.llm.openai, "model_name", "") or ""
    if fam == "exo":
        return "exo", getattr(config.exo, "model_name", "") or ""
    if fam == "mlx":
        if not Environment.is_oss_local_provider(config.llm.provider):
            return _provider_model_for_main(config)
        return "mlx", (
            getattr(orch, "mlx_model", "") or getattr(config.llm.mlx, "hf_llm_model_id", "") or ""
        )
    return _provider_model_for_main(config)


def _provider_model_for_subagent(
    agent_spec: AgentSpec,
    config: Any,
    parent_provider: str,
    parent_model_id: str,
) -> tuple[str, str]:
    """Return ``(provider, model_id)`` for a library subagent.

    Mirrors :func:`_resolve_subagent_model` so the vision-capability decision
    matches the model that will actually be instantiated.  ``parent_*`` is the
    orchestrator's provider/model (what ``inherit``/``None`` agents receive).
    """
    from utilities.environment import Environment

    fam_raw = agent_spec.subagent_llm_family
    fam = fam_raw.strip().lower() if isinstance(fam_raw, str) and fam_raw.strip() else None

    if fam is None:
        if agent_spec.model_override:
            return _split_provider_model(agent_spec.model_override)
        return parent_provider, parent_model_id
    if fam == "inherit":
        return parent_provider, parent_model_id
    if fam == "frontier":
        return "anthropic", getattr(config.llm.anthropic, "model_name", "") or ""
    if fam == "mlx":
        if not Environment.is_oss_local_provider():
            return parent_provider, parent_model_id
        return "mlx", (
            (agent_spec.mlx_model_id or "").strip()
            or getattr(config.llm.mlx, "hf_llm_model_id", "")
            or ""
        )
    if fam == "exo":
        return "exo", getattr(config.exo, "model_name", "") or ""
    if fam == "custom":
        if agent_spec.model_override:
            return _split_provider_model(agent_spec.model_override)
        return parent_provider, parent_model_id
    return parent_provider, parent_model_id


def _apply_macos_vision_variant(
    tools: list[Any], conn: Any,
) -> list[Any]:
    """Swap the text ``read_screen`` for the vision combo + add screenshot tool.

    Uses the pre-wrapped variants stashed on the macos-native connection by the
    loader.  Returns *tools* unchanged when the stash is missing (e.g. the
    connection failed to load the vision variants).
    """
    stash = getattr(conn, "macos_native_vision", None) if conn is not None else None
    if not stash:
        return tools
    # Flip the toolkit into vision mode so the get_screen_controls AX-disabled
    # fallback hint tells the model read_screen also returns a screenshot to
    # look at (instead of framing it as text-only OCR).
    toolkit = stash.get("toolkit")
    if toolkit is not None:
        try:
            toolkit.vision_mode = True
        except Exception:
            pass
    vision_read = stash.get("read_screen")
    out = [
        vision_read if (vision_read is not None and t.name == "read_screen") else t
        for t in tools
    ]
    present = {t.name for t in out}
    for extra in stash.get("extra", []) or []:
        if extra.name not in present:
            out.append(extra)
            present.add(extra.name)
    return out


def _build_subagents_from_library(
    mcp_mgr: Any,
    standard_tools: list[Any],
    model: Any,
    backend: Any,
    ask_user_tools: list[Any] | None = None,
    *,
    pw_pool: Any = None,
    memory_middleware: list[Any] | None = None,
    activity_tools: list[Any] | None = None,
    subagent_vision_resolver: Any = None,
    session_id: str | None = None,
) -> tuple[Optional[list[dict[str, Any]]], set[str]]:
    """Build ``CompiledSubAgent`` specs from every agent in the library.

    Each agent becomes a ``subagent_type`` the LLM can spawn via the ``task``
    tool.  The agent graph is compiled here (with the standard middleware
    stack) and wrapped in :class:`StreamingSubagentRunnable` so that
    intermediate tool calls / results are relayed to the chat in real time.

    Agents whose required MCP servers aren't connected are skipped.
    Agents with no declared MCP tools are skipped (they would duplicate the
    built-in general-purpose subagent).

    Returns ``(subagent_list_or_None, set_of_claimed_server_ids)``.
    """
    from langchain.agents import create_agent
    from langchain.agents.middleware import TodoListMiddleware
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from deepagents.middleware.summarization import (
        SummarizationMiddleware,
        compute_summarization_defaults,
    )

    from backend.agent_library import get_agent_system_prompt, list_agents
    from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware
    from backend.prompts import STRUCTURED_SUMMARY_PROMPT, STRUCTURED_SUMMARY_PROMPT_LITE
    from backend.streaming_subagent import StreamingSubagentRunnable
    from utilities.environment import Environment

    subagents: list[dict[str, Any]] = []
    claimed_server_ids: set[str] = set()

    connections = mcp_mgr.connections
    connected_ids = {sid for sid, conn in connections.items() if conn.connected}

    for agent_spec in list_agents():
        if agent_spec.name in _RESERVED_SUBAGENT_NAMES:
            logger.debug("Skipping agent %r — reserved subagent name", agent_spec.name)
            continue

        required_ids = set(agent_spec.tools)

        if not required_ids:
            logger.debug(
                "Skipping agent %r — no MCP tools declared (would duplicate general-purpose)",
                agent_spec.name,
            )
            continue

        if not required_ids.intersection(connected_ids):
            continue

        from backend.mcp_manager import dedupe_tool_names

        agent_mcp_tools = dedupe_tool_names(
            (sid, connections[sid].tools)
            for sid in required_ids
            if sid in connections and connections[sid].connected
        )
        if not agent_mcp_tools:
            continue

        # For a vision-capable subagent model, swap the macos-native text
        # read_screen for the text+image combo and expose capture_app_screenshot
        # so the VLM actually receives screenshots. Text-only models keep the
        # text-only variant (images would be useless / dropped by the server).
        if "macos-native" in required_ids and subagent_vision_resolver is not None:
            try:
                wants_vision = bool(subagent_vision_resolver(agent_spec))
            except Exception:
                wants_vision = False
            if wants_vision:
                agent_mcp_tools = _apply_macos_vision_variant(
                    agent_mcp_tools, connections.get("macos-native"),
                )
                logger.info(
                    "macos-native: vision read_screen enabled for subagent %r",
                    agent_spec.name,
                )

        system_prompt = get_agent_system_prompt(agent_spec.name) or ""
        claimed_server_ids.update(required_ids)

        subagent_model = _resolve_subagent_model(agent_spec, model)

        summ_defaults = compute_summarization_defaults(subagent_model)
        # Use the lite summary prompt when this subagent's model is on
        # an OSS-local provider (mlx / exo).  See
        # ``backend.prompts.STRUCTURED_SUMMARY_PROMPT_LITE`` for what the
        # lite version drops.
        sa_fam = (agent_spec.subagent_llm_family or "").strip().lower()
        sa_is_local = (
            sa_fam in ("mlx", "exo")
            or (sa_fam in ("", "inherit") and Environment.is_oss_local_provider())
        )
        sa_summary_prompt = (
            STRUCTURED_SUMMARY_PROMPT_LITE if sa_is_local else STRUCTURED_SUMMARY_PROMPT
        )
        middleware: list[Any] = [
            TodoListMiddleware(),
            FilesystemMiddleware(backend=backend),
            SummarizationMiddleware(
                model=subagent_model,
                backend=backend,
                trigger=summ_defaults["trigger"],
                keep=summ_defaults["keep"],
                summary_prompt=sa_summary_prompt,
                trim_tokens_to_summarize=None,
                truncate_args_settings=summ_defaults["truncate_args_settings"],
            ),
        ]
        if memory_middleware:
            middleware.extend(memory_middleware)
        caching_mw = _make_prompt_caching_middleware(subagent_model)
        if caching_mw:
            middleware.append(caching_mw)
        middleware.append(PatchToolCallsMiddleware())

        from backend.safety_middleware import (
            ExecutePathSafetyMiddleware,
            HighRiskExecuteFlaggerMiddleware,
        )
        middleware.append(ExecutePathSafetyMiddleware())
        middleware.append(HighRiskExecuteFlaggerMiddleware())

        if "playwright-mcp" in required_ids:
            middleware.insert(0, PlaywrightSnapshotPruningMiddleware())
            logger.info(
                "Playwright snapshot pruning enabled for %r",
                agent_spec.name,
            )

        if "macos-native" in required_ids:
            from agents.computer_voyager import ScreenControlPruningMiddleware
            middleware.insert(0, ScreenControlPruningMiddleware())
            logger.info(
                "Screen control pruning enabled for %r",
                agent_spec.name,
            )

        # ReAct shim for text-only on-device subagent models (e.g. an
        # MLX subagent whose template lacks native tool support).
        # Self-gated by ``_maybe_react_shim`` — no-op for tool-capable
        # models.
        sa_react_shim = _maybe_react_shim(subagent_model)
        if sa_react_shim is not None:
            middleware.insert(0, sa_react_shim)
            logger.info(
                "ReAct shim enabled for subagent %r (model=%s)",
                agent_spec.name, type(subagent_model).__name__,
            )

        # Context-window safety net: clip the request to fit the model's
        # input budget after every other middleware has run.  Appended last
        # so it executes innermost (closest to the model).  No-op for
        # big-context providers.
        sa_ctx_trunc = _maybe_context_truncation(subagent_model)
        if sa_ctx_trunc is not None:
            middleware.append(sa_ctx_trunc)
            logger.info(
                "Context truncation enabled for subagent %r (model=%s, budget=%d tok)",
                agent_spec.name,
                type(subagent_model).__name__,
                sa_ctx_trunc._budget,
            )

        extra_tools: list[Any] = []

        # Meta-agents author other agents / triggers / MCP servers and need
        # the full management surface even when invoked via the orchestrator's
        # ``task`` tool — without these their system prompts (which instruct
        # them to call ``create_agent_config`` / ``create_trigger`` / …) have
        # no matching tools and the model improvises by writing files
        # directly to the real filesystem (which the path-safety guard then
        # correctly blocks).
        _META_AGENT_NAMES = {"trigger-builder-agent", "mcp-builder-agent", "schedule-builder-agent"}
        if agent_spec.name in _META_AGENT_NAMES:
            from backend.agent_management_tools import build_management_tools
            from backend.mcp_builder_tools import build_mcp_builder_tools
            from backend.schedule_tools import build_schedule_tools
            from backend.trigger_tools import build_trigger_tools

            extra_tools.extend(build_management_tools(mcp_mgr))
            extra_tools.extend(build_mcp_builder_tools())
            # ``agent_name`` gates privileged trigger types (http/git/shell)
            # to the trigger-builder-agent only; the mcp-builder-agent gets
            # them too here, which is fine — it has no path that calls
            # ``create_trigger`` so the extra capability is unused.
            extra_tools.extend(build_trigger_tools(agent_name=agent_spec.name))
            extra_tools.extend(build_schedule_tools())
            logger.info(
                "Meta-agent %r: attached management/trigger/schedule/mcp_builder tools",
                agent_spec.name,
            )

        subagent_tools = (
            agent_mcp_tools
            + list(standard_tools)
            + list(activity_tools or [])
            + extra_tools
            + list(ask_user_tools or [])
        )
        _apply_universal_loop_guard(
            subagent_tools,
            scope=f"subagent:{agent_spec.name}",
            session_id=session_id,
        )
        _insert_repeated_thought_guard(
            middleware, scope=f"subagent:{agent_spec.name}"
        )
        graph = create_agent(
            subagent_model,
            system_prompt=system_prompt,
            tools=subagent_tools,
            middleware=middleware,
            name=agent_spec.name,
        )

        agent_pw_pool = pw_pool if "playwright-mcp" in required_ids else None
        subagents.append({
            "name": agent_spec.name,
            "description": agent_spec.description,
            "runnable": StreamingSubagentRunnable(
                graph, agent_spec.name, pw_pool=agent_pw_pool,
            ),
        })

    if subagents:
        logger.info(
            "Subagents loaded: %s",
            [s["name"] for s in subagents],
        )
    return (subagents or None, claimed_server_ids)


class Session:
    """A running agent session."""

    def __init__(
        self,
        session_id: str,
        agent_name: Optional[str],
        graph: Any,
        tool_set: Any = None,
        checkpointer: Any = None,
        _sqlite_conn: Any = None,
        schedule_id: Optional[str] = None,
        trigger_source: Optional[str] = None,
        trigger_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        chain_depth: int = 0,
        llm_provider: str = "",
    ) -> None:
        self.id = session_id
        self.agent_name = agent_name
        self.graph = graph
        self.tool_set = tool_set
        self._checkpointer = checkpointer
        self._sqlite_conn = _sqlite_conn
        self.title = "New Session"
        self.message_count = 0
        self.tools_used: list[str] = []
        self.schedule_id = schedule_id
        self.trigger_source = trigger_source
        self.trigger_id = trigger_id
        self.parent_session_id = parent_session_id
        self.chain_depth = chain_depth
        # Provider used to construct the LLM for this session.  Stamped at
        # creation so per-turn privacy checks can inspect it without
        # re-reading the config (the user may change the config mid-session).
        self.llm_provider = llm_provider
        self.memory_inject = False
        self.recursion_limit: int = 10000
        self.live_output_queue: asyncio.Queue = asyncio.Queue()
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        # Run metrics
        self.status: str = "idle"
        self.finished_at: Optional[datetime] = None
        self.duration_ms: Optional[int] = None
        self.model: Optional[str] = None
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.error: Optional[str] = None
        self.error_code: Optional[str] = None
        # MLX throughput accumulators (only populated for in-process MLX).
        # We keep raw token counts plus elapsed time so the derived averages
        # exposed via ``to_info`` are token-weighted rather than naive means.
        self.prefill_tokens_total: int = 0
        self.cache_tokens_total: int = 0
        self.gen_tokens_total: int = 0
        self.prefill_time_total: float = 0.0
        self.gen_time_total: float = 0.0
        self.peak_memory_gb: float = 0.0
        # End-of-run evaluation summary (set by backend.eval_runner).
        self.eval_status: Optional[str] = None
        self.eval_overall_score: Optional[float] = None
        self.eval_pass_count: Optional[int] = None
        self.eval_total: Optional[int] = None

    def accumulate_mlx_stats(self, stats: dict) -> None:
        """Fold a single turn's MLX ``response_metadata`` stats into the
        session-level throughput accumulators.  Safe to call with partial or
        empty dicts (non-MLX providers simply contribute nothing)."""
        if not stats:
            return
        prefilled = int(stats.get("tokens_prefilled") or 0)
        cached = int(stats.get("tokens_from_cache") or 0)
        gen_tokens = int(stats.get("generation_tokens") or 0)
        prompt_tps = float(stats.get("prompt_tps") or 0.0)
        gen_tps = float(stats.get("generation_tps") or 0.0)
        peak = float(stats.get("peak_memory_gb") or 0.0)

        self.prefill_tokens_total += prefilled
        self.cache_tokens_total += cached
        self.gen_tokens_total += gen_tokens
        if prefilled and prompt_tps > 0:
            self.prefill_time_total += prefilled / prompt_tps
        if gen_tokens and gen_tps > 0:
            self.gen_time_total += gen_tokens / gen_tps
        if peak > self.peak_memory_gb:
            self.peak_memory_gb = peak

    def _throughput_fields(self) -> dict:
        """Derive token-weighted TIPS/TOPS, KV cache hit ratio and peak GPU
        memory from the accumulators.  Returns ``None`` values when no MLX
        data was collected (e.g. omlx/cloud sessions)."""
        avg_prefill_tps = (
            round(self.prefill_tokens_total / self.prefill_time_total, 1)
            if self.prefill_time_total > 0 else None
        )
        avg_generation_tps = (
            round(self.gen_tokens_total / self.gen_time_total, 1)
            if self.gen_time_total > 0 else None
        )
        cache_total = self.cache_tokens_total + self.prefill_tokens_total
        cache_hit_ratio = (
            round(self.cache_tokens_total / cache_total, 3)
            if cache_total > 0 else None
        )
        return {
            "avg_prefill_tps": avg_prefill_tps,
            "avg_generation_tps": avg_generation_tps,
            "cache_hit_ratio": cache_hit_ratio,
            "peak_memory_gb": round(self.peak_memory_gb, 3) if self.peak_memory_gb else None,
        }

    def to_info(self) -> SessionInfo:
        return SessionInfo(
            id=self.id,
            agent_name=self.agent_name,
            title=self.title,
            message_count=self.message_count,
            tools_used=self.tools_used,
            schedule_id=self.schedule_id,
            trigger_source=self.trigger_source,
            trigger_id=self.trigger_id,
            parent_session_id=self.parent_session_id,
            chain_depth=self.chain_depth,
            created_at=self.created_at,
            updated_at=self.updated_at,
            status=self.status,
            finished_at=self.finished_at,
            duration_ms=self.duration_ms,
            llm_provider=self.llm_provider or None,
            model=self.model,
            input_tokens=self.input_tokens or None,
            output_tokens=self.output_tokens or None,
            estimated_cost_usd=self.estimated_cost_usd or None,
            error=self.error,
            error_code=self.error_code,
            eval_status=self.eval_status,
            eval_overall_score=self.eval_overall_score,
            eval_pass_count=self.eval_pass_count,
            eval_total=self.eval_total,
            **self._throughput_fields(),
        )

    async def close(self) -> None:
        if self.tool_set is not None:
            await self.tool_set.close()
        if self._sqlite_conn is not None:
            try:
                await self._sqlite_conn.close()
            except Exception:
                logger.debug("Error closing sqlite connection", exc_info=True)
            self._sqlite_conn = None

    def save_meta(self) -> None:
        meta_path = _sessions_dir() / f"{self.id}.json"
        meta_path.write_text(self.to_info().model_dump_json(indent=2), encoding="utf-8")

    async def save_meta_async(self) -> None:
        await asyncio.to_thread(self.save_meta)


_SESSION_IDLE_TIMEOUT_SECS = 30 * 60  # 30 minutes
_MEMORY_EXTRACTION_INTERVAL = 3
_MAX_SESSION_HISTORY = 50


# Per-model lock keyed by ``model_name`` so concurrent session starts that
# all want the same oMLX model don't pile up parallel ``omlx serve --model``
# spawns.  The first caller does the work; later callers wait on the same
# lock and then return immediately because the model is now loaded.
_omlx_model_locks: dict[str, asyncio.Lock] = {}


async def _ensure_omlx_model_loaded(config: "AppConfig") -> None:
    """Ensure the oMLX server is running and the configured model is loaded.

    Called once at session-start time so the user never hits a cryptic
    404 or connection error mid-chat.  Strategy:

    1. If the server isn't running, start it via :func:`astart` (which
       also performs one-time admin provisioning — set API key, point
       ``model_dirs`` at the HF cache, enable ``skip_api_key_verification``).
    2. Resolve the configured model name to oMLX's short id, refreshing
       provisioning if it isn't found yet.
    3. If the resolved model isn't already loaded, call :func:`aload_model`
       which dynamically loads it via the HTTP API (no restart).

    Concurrent calls for the same model coalesce via
    :data:`_omlx_model_locks` to avoid pile-ups when multiple sessions
    start at once.
    """
    from backend.omlx_provisioner import (
        _resolve_omlx_model_id, adopt_existing_admin_key, afetch_status,
        aload_model, astart,
    )

    model_name = (config.omlx.model_name or "").strip()
    if not model_name:
        return

    lock = _omlx_model_locks.setdefault(model_name, asyncio.Lock())
    async with lock:
        status = await afetch_status(config.omlx)

        # Self-heal the admin key even when the server is already running.
        # astart() (which normally adopts/provisions the key) is skipped for a
        # reachable server, so without this an externally-started oMLX leaves
        # Otto's admin_api_key blank and later /admin/api/* calls 400.
        if status.get("reachable"):
            adopt_existing_admin_key(config.omlx)

        if not status.get("reachable"):
            logger.info(
                "oMLX: server not reachable at %s — auto-starting",
                config.omlx.effective_base_url,
            )
            start_job = await astart(config.omlx)
            await _wait_for_omlx_job(
                start_job.id, timeout=120.0, description="start the oMLX server",
            )
            status = await afetch_status(config.omlx)
            if not status.get("reachable"):
                raise RuntimeError(
                    "oMLX server still not reachable after start attempt. "
                    "Check the oMLX setup screen for details."
                )

        resolved = await _resolve_omlx_model_id(config.omlx, model_name)
        loaded_ids = [m["id"] for m in (status.get("models") or [])]
        if resolved is not None and resolved in loaded_ids:
            logger.info(
                "oMLX: model '%s' (oMLX id '%s') already loaded — skipping auto-load",
                model_name, resolved,
            )
            return

        logger.info(
            "oMLX: loading model '%s' via HTTP /v1/models/<id>/load",
            model_name,
        )
        job = await aload_model(config.omlx, model_name)
        await _wait_for_omlx_job(
            job.id, timeout=600.0,
            description=f"load model '{model_name}'",
        )

        verify = await afetch_status(config.omlx)
        verify_ids = [m["id"] for m in (verify.get("models") or [])]
        resolved = await _resolve_omlx_model_id(config.omlx, model_name)
        if resolved is not None and resolved in verify_ids:
            logger.info(
                "oMLX: auto-load of '%s' (oMLX id '%s') completed",
                model_name, resolved,
            )
            return
        raise RuntimeError(
            f"oMLX auto-load reported success but '{model_name}' is "
            f"not present in /v1/models. Loaded: {verify_ids or '(none)'}. "
            "Try loading the model manually from the oMLX setup screen."
        )


async def _wait_for_omlx_job(job_id: str, *, timeout: float, description: str) -> None:
    """Block until an oMLX job reports done/error or *timeout* elapses."""
    from backend.omlx_provisioner import get_job
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(2.0)
        j = get_job(job_id)
        if j is None:
            return
        if j.status == "done":
            return
        if j.status == "error":
            raise RuntimeError(
                f"oMLX failed to {description}: {j.error or '(no detail)'}"
            )
    raise RuntimeError(
        f"oMLX did not {description} within {timeout / 60:.0f} minutes. "
        "Use the oMLX setup screen to load the model manually."
    )


class SessionManager:
    """Manages active and historical sessions."""

    def __init__(self) -> None:
        self._active: dict[str, Session] = {}
        self._eviction_task: Optional[asyncio.Task] = None

    async def _build_graph(
        self,
        config: AppConfig,
        agent_name: Optional[str],
        session_id: str,
        checkpointer: Any,
        *,
        is_scheduled_run: bool = False,
        schedule_id: Optional[str] = None,
        live_output_queue: Optional[asyncio.Queue] = None,
    ) -> tuple[Any, Any, asyncio.Queue]:
        """Build the agent graph and MCP tool set.

        When *agent_name* is set the user explicitly chose an agent via the
        ``/`` slash-command — the agent runs directly with its own MCP tools.

        When *agent_name* is ``None`` (general-purpose mode), every agent in
        the library is registered as a subagent so the LLM can delegate via
        the ``task`` tool.  MCP tools claimed by subagents are excluded from
        the main agent to avoid duplication.

        All agents (main and subagents) receive the standard research tools
        (web researcher, doc researcher, doc reader, wikipedia).

        Returns ``(graph, mcp_mgr, live_output_queue)``.
        """
        if live_output_queue is None:
            live_output_queue = asyncio.Queue()
        from backend.agent_library import get_agent, get_agent_system_prompt
        from backend.mcp_manager import MCPManager

        t_graph_start = time.monotonic()
        logger.info("[build_graph] START session=%s agent=%s", session_id, agent_name)

        system_prompt: Optional[str] = None
        tool_mcp_ids: set[str] = set()

        if agent_name:
            agent_spec = get_agent(agent_name)
            if agent_spec:
                system_prompt = get_agent_system_prompt(agent_name)
                tool_mcp_ids.update(agent_spec.tools)

        if not system_prompt:
            from deep_agent.prompt import build_orchestrator_prompt
            from utilities.environment import Environment

            use_lite = Environment.use_lite_orchestrator_prompt()
            system_prompt = build_orchestrator_prompt([], lite=use_lite)
            logger.info(
                "[build_graph] orchestrator prompt mode=%s (provider=%s, da_provider=%s)",
                "lite" if use_lite else "full",
                Environment.get_llm_provider(),
                Environment.get_deep_agent_llm_provider() or "(inherit)",
            )

        mcp_mgr = MCPManager()
        _os_label = platform_label()
        enabled_servers = [
            s for s in config.mcp_servers
            if s.enabled
            and (not s.requires_os or s.requires_os == _os_label)
            and (not tool_mcp_ids or s.id in tool_mcp_ids)
        ]
        logger.info("[build_graph] connecting %d MCP servers", len(enabled_servers))
        t_mcp = time.monotonic()
        try:
            await asyncio.shield(
                mcp_mgr.connect_all(enabled_servers, skip_process_start=True)
            )
        except Exception as exc:
            logger.warning("MCP connect_all error — continuing with available tools: %s", exc)
        logger.info("[build_graph] MCP connect_all done in %.1fs", time.monotonic() - t_mcp)

        from backend.agent_management_tools import build_management_tools
        from backend.ask_user_tools import build_ask_user_tools
        from backend.file_tools import build_file_tools
        from backend.activity_tools import build_activity_tools
        from backend.recap_tools import build_recap_tools
        from backend.ambient_tools import build_ambient_tools
        from backend.mcp_builder_tools import build_mcp_builder_tools
        from backend.memory_extraction import MemoryExtractionMiddleware
        from backend.schedule_tools import build_schedule_tools
        from backend.trigger_tools import build_trigger_tools
        from backend.exo_tools import build_exo_tools
        from backend.omlx_tools import build_omlx_tools
        from backend.privacy_tools import build_privacy_tools
        from backend.settings_tools import build_settings_tools
        from backend.spawn_tools import build_spawn_tools
        from backend.session_search_tools import build_session_search_tools
        from deep_agent.model_factory import create_llm, create_mlx_vlm, supports_vision
        from deepagents import create_deep_agent

        # oMLX: ensure the configured model is actually loaded in the server
        # before we wire up the LLM client.  The server starts empty — a
        # model must be explicitly loaded (via CLI or a restart) before any
        # /v1/chat/completions request can succeed.  We do this here so the
        # user gets a clear progress log instead of a cryptic 404 mid-chat.
        if config.llm.provider == "omlx" and config.omlx.model_name:
            await _ensure_omlx_model_loaded(config)

        # create_llm for MLX loads (and on a cache miss would download) the
        # model weights — must run in a thread, never on the event loop.
        llm = await asyncio.to_thread(create_llm, config.llm.provider)

        # Stamp .profile on local-server models (oMLX, exo) so that
        # compute_summarization_defaults uses the fraction-based trigger
        # (85 % of budget) instead of the 170 k-token fallback, which
        # silently exceeds these models' actual context windows and causes
        # "prompt too long" errors.  ChatMLXText sets its own profile in
        # __init__; cloud models (Anthropic, OpenAI) are handled by deepagents.
        if config.llm.provider == "omlx":
            _budget = int(getattr(config.omlx, "max_context_window", 0) or 131072)
            try:
                llm.profile = {"max_input_tokens": _budget}
            except Exception:
                object.__setattr__(llm, "profile", {"max_input_tokens": _budget})

        # Resolve the orchestrator model.  When orchestrator.llm_family is
        # "follow_main" (the default) orch_llm == llm.  Otherwise it is the
        # provider specified by the orchestrator setting (e.g. "anthropic" for
        # "frontier").  Using orch_llm as the parent_model for subagents ensures
        # that "inherit" agents truly share the orchestrator's provider instead
        # of always falling back to the main chat provider.
        orch_llm = await asyncio.to_thread(_build_orch_llm_sync, config, llm)

        # Resolve a vision model for file tools.  When the main provider is
        # text-only, view_image / load_image_from_url will use a dedicated VLM
        # to describe images as text instead of forwarding raw image blocks the
        # text model cannot interpret.
        #
        # Regardless of the main provider, we always check the MLX VLM path
        # (create_mlx_vlm with provider="mlx") so that HF_VLM_MODEL_ID acts as
        # a universal vision fallback — even when the main provider is omlx, exo,
        # or a text-only local MLX model.  For cloud providers (Anthropic, OpenAI)
        # that already handle images natively, supports_vision returns True and we
        # skip this entirely, forwarding raw image blocks as before.
        _main_provider = config.llm.provider
        # Resolve the model id for the *active* provider.  ``config.llm.mlx``
        # carries a non-empty default model id even when the provider is oMLX
        # or exo, so a naive ``mlx or omlx or exo`` fallback would wrongly pick
        # the stale MLX id and misclassify vision capability.
        if _main_provider == "omlx":
            _main_model_id = getattr(config.omlx, "model_name", "") or ""
        elif _main_provider == "exo":
            _main_model_id = getattr(config.exo, "model_name", "") or ""
        elif _main_provider == "mlx":
            _main_model_id = getattr(config.llm.mlx, "hf_llm_model_id", "") or ""
        else:
            _main_model_id = ""
        _main_supports_vision = supports_vision(_main_provider, _main_model_id)
        if not _main_supports_vision:
            # Always use provider="mlx" here so create_mlx_vlm checks
            # HF_VLM_MODEL_ID regardless of the actual main provider.
            _vision_llm = await asyncio.to_thread(create_mlx_vlm, "mlx", llm)
            # create_mlx_vlm returns llm itself when HF_VLM_MODEL_ID is unset.
            _file_vision_llm = _vision_llm if _vision_llm is not llm else None
            if _file_vision_llm is not None:
                _vlm_name = getattr(_file_vision_llm, "model", None) or type(_file_vision_llm).__name__
                logger.info(
                    "Vision: dedicated VLM '%s' will describe images (main provider '%s' is text-only)",
                    _vlm_name, _main_provider,
                )
            else:
                logger.info(
                    "Vision: main provider '%s' is text-only and no HF_VLM_MODEL_ID is set — "
                    "image blocks forwarded as-is (model may not see them)",
                    _main_provider,
                )
        else:
            _file_vision_llm = None
            logger.info(
                "Vision: native image blocks forwarded to %s model '%s'",
                _main_provider, _main_model_id or "(default)",
            )

        # Document tools extract embedded images only when the session can
        # actually interpret them — either the main model is vision-capable or
        # a dedicated VLM is available to describe them via view_image.
        _vision_available = _main_supports_vision or (_file_vision_llm is not None)

        files_dir = _session_files_dir(session_id)
        backend = _make_backend(files_dir)

        # --- Playwright browser isolation pool ---
        # When Playwright MCP is connected, create a pool of ephemeral
        # browser instances so concurrent subagents each get their own
        # browser instead of fighting over a single shared one.
        pw_pool = await _maybe_create_pw_pool(mcp_mgr)

        standard_tools = _build_standard_tools(
            orch_llm,
            files_dir=files_dir,
            backend=backend,
            extract_images=_vision_available,
        )
        management_tools = build_management_tools(mcp_mgr)
        # mcp_builder_tools let the agent author its own MCP servers at
        # runtime — see backend.mcp_builder for the generation pipeline
        # and backend.credential_vault for the secret-handling story.
        mcp_builder_tools = build_mcp_builder_tools()
        schedule_tools = build_schedule_tools()
        trigger_tools = build_trigger_tools(agent_name=agent_name)
        file_tools = build_file_tools(files_dir, vision_llm=_file_vision_llm)
        # Activity tools query the local activity timeline DB.  Always
        # included — the tools themselves return "no data" when the
        # tracker is disabled, so the model can recognise the situation
        # rather than thinking the tool doesn't exist.
        activity_tools = build_activity_tools() if config.activity.enabled else []
        recap_tools = build_recap_tools()
        ambient_tools = build_ambient_tools()
        ask_user_tools = build_ask_user_tools()
        # settings_tools let the agent inspect and adjust orchestrator,
        # generation, memory, and MCP server settings at runtime.
        settings_tools = build_settings_tools()
        privacy_tools = build_privacy_tools()
        # spawn_tools let the orchestrator hand off the current request
        # to a fresh session — typically used after building new MCP
        # tools or sub-agents that aren't bound to this graph yet.
        # Closes over session_id so the tool knows which session is the
        # parent without needing LangGraph runtime context.
        spawn_tools = build_spawn_tools(session_id)
        session_search_tools = build_session_search_tools()
        # exo cluster tools are read-only diagnostics — cluster lifecycle
        # (start/stop/provision, add/remove remotes) is a human action via
        # the ExoPage UI.  We only attach the tools when the user has
        # opted in via Settings *and* the local daemon is actually
        # running, so the agent never sees them on a cold install or
        # while the cluster is down.
        exo_tools: list = []
        if config.exo.enabled:
            from backend.exo_cli import is_running as _exo_is_running
            if _exo_is_running(config.exo.api_port):
                exo_tools = build_exo_tools()
            else:
                logger.info(
                    "exo enabled but daemon not running on :%d — skipping exo tools",
                    config.exo.api_port,
                )

        # oMLX local server tools — same opt-in + reachability gating as
        # exo. We probe the configured port instead of a pidfile because
        # oMLX may be running under ``brew services`` (managed by
        # launchd) or under the macOS GUI app, neither of which exposes
        # a pidfile we control.
        omlx_tools: list = []
        if config.omlx.enabled:
            try:
                import socket as _sock
                with _sock.socket(_sock.AF_INET, _sock.SOCK_STREAM) as s:
                    s.settimeout(0.25)
                    _omlx_running = s.connect_ex(("127.0.0.1", config.omlx.api_port)) == 0
            except OSError:
                _omlx_running = False
            if _omlx_running:
                omlx_tools = build_omlx_tools()
            else:
                logger.info(
                    "omlx enabled but server not listening on :%d — skipping omlx tools",
                    config.omlx.api_port,
                )

        # Build memory middleware once so it can be shared with sub-agents.
        # The two layers are independent: a user can enable either, both, or
        # neither.  ``inject_enabled`` is a legacy combined flag kept for
        # backward compatibility — it implicitly enables both layers.
        memory_middleware: list[Any] = []
        l1_enabled = config.memory.effective_inject_on_session_start
        l2_enabled = config.memory.effective_inject_realtime

        if l1_enabled:
            from backend.memory import _index_path
            idx = _index_path()
            if idx.exists():
                from deepagents.backends.filesystem import (
                    FilesystemBackend as _FSBackend,
                )
                from deepagents.middleware.memory import (
                    MemoryMiddleware,
                )
                # MEMORY.md lives on the real filesystem outside the session's
                # virtual root.  The session backend (_SymlinkAwareBackend,
                # virtual_mode=True) deliberately rejects real /Users/… paths,
                # so we use a plain FilesystemBackend here instead.
                _memory_fs_backend = _FSBackend(virtual_mode=False)
                memory_middleware.append(
                    MemoryMiddleware(
                        backend=_memory_fs_backend,
                        sources=[str(idx)],
                    )
                )
                logger.info(
                    "[build_graph] memory L1 (MEMORY.md) enabled for session=%s",
                    session_id,
                )
            else:
                logger.info(
                    "[build_graph] memory L1 skipped — no MEMORY.md index yet, session=%s",
                    session_id,
                )

        if l2_enabled:
            from backend.memory_relevance import (
                MemoryRelevanceMiddleware,
            )
            memory_middleware.append(
                MemoryRelevanceMiddleware(
                    memory_cfg=config.memory,
                    llm_provider=config.llm.provider,
                    session_id=session_id,
                )
            )
            logger.info(
                "[build_graph] memory L2 (per-turn relevance) enabled for session=%s",
                session_id,
            )

        if not (l1_enabled or l2_enabled):
            logger.info(
                "[build_graph] memory injection disabled for session=%s",
                session_id,
            )

        subagents = None
        main_has_playwright = False

        # ``graph_llm`` is the model used for the main agent graph, the
        # orchestrator caching middleware, and memory extraction.  It equals
        # ``orch_llm`` in both GP and direct-agent mode.  When the orchestrator
        # model could not be created (e.g. MLX weights not yet downloaded),
        # ``orch_llm`` already fell back to the main chat LLM (``llm``).
        graph_llm: Any = orch_llm

        # Vision capability of the model that actually runs the main graph
        # (the orchestrator / direct agent).  Drives the macos-native
        # read_screen text→image-combo swap for the main agent in BOTH
        # direct-agent and GP mode.  Library subagents are resolved separately
        # below (their model may differ from the orchestrator's).
        _orch_provider, _orch_model_id = _provider_model_for_orchestrator(config)
        try:
            _orch_supports_vision = supports_vision(_orch_provider, _orch_model_id)
        except Exception:
            _orch_supports_vision = False

        if agent_name:
            # Direct agent mode — agent gets its own MCP tools directly.
            # graph_llm is already set to orch_llm above, which correctly
            # reflects the active provider (e.g. "frontier") and falls back to
            # the main chat LLM when the orchestrator model can't be created.
            agent_extra_tools: list[Any] = []

            # When the direct agent's model is vision-capable, swap the
            # macos-native text read_screen for the text+image combo and add
            # capture_app_screenshot so the VLM actually receives screenshots.
            # No-op when macos-native isn't connected or no vision stash exists.
            mcp_tools = mcp_mgr.get_all_tools()
            if _orch_supports_vision:
                mcp_tools = _apply_macos_vision_variant(
                    mcp_tools, mcp_mgr.connections.get("macos-native"),
                )
                logger.info(
                    "macos-native: vision read_screen enabled for direct agent %r",
                    agent_name,
                )

            all_tools = (
                mcp_tools
                + standard_tools
                + management_tools
                + mcp_builder_tools
                + spawn_tools
                + session_search_tools
                + schedule_tools
                + trigger_tools
                + exo_tools
                + omlx_tools
                + settings_tools
                + privacy_tools
                + file_tools
                + activity_tools
                + recap_tools
                + ambient_tools
                + ask_user_tools
                + agent_extra_tools
            )
            # In direct-agent mode there are no sub-agents to claim MCP tools,
            # so the orchestrator owns every connected tool.  ``pw_pool`` is
            # created iff the ``playwright-mcp`` server is actually connected.
            main_has_playwright = pw_pool is not None

            # Build the properly-configured GP subagent for direct-agent mode.
            # Without this, ``create_deep_agent`` would add a default GP
            # subagent that (a) inherits ``interrupt_on={"execute": True}``
            # from the parent — causing HITL interrupts inside ``task`` that
            # hang in scheduled/unattended runs — and (b) lacks the
            # StreamingSubagentRunnable wrapper needed for live UI feedback.
            # Passing a pre-compiled ``CompiledSubAgent`` here causes
            # ``create_deep_agent`` to skip its built-in GP subagent entirely.
            gp_subagent = _build_gp_subagent(
                model=orch_llm,
                tools=all_tools,
                backend=backend,
                pw_pool=pw_pool if main_has_playwright else None,
                memory_middleware=memory_middleware or None,
                session_id=session_id,
            )
            # Also register the current named agent as a self-subagent so the
            # orchestrator can delegate subtasks to it via
            # ``task(subagent_type="<agent_name>")``.  This is critical for
            # scheduled runs: without it the assigned agent is only the
            # top-level caller and inner ``task`` calls can only reach
            # ``general-purpose``.
            named_subagent = _build_named_agent_subagent(
                agent_name=agent_name,
                agent_spec=agent_spec,
                model=orch_llm,
                tools=all_tools,
                backend=backend,
                pw_pool=pw_pool if main_has_playwright else None,
                memory_middleware=memory_middleware or None,
                session_id=session_id,
            )
            if named_subagent is not None:
                subagents = [gp_subagent, named_subagent]
            else:
                subagents = [gp_subagent]
        else:
            # General-purpose mode — register library agents as subagents
            # and keep only unclaimed MCP tools on the main agent.
            # Pass orch_llm (not llm) so "inherit" subagents receive the
            # orchestrator's provider as their parent model, matching the
            # orchestratorChip() display in the frontend.
            # Resolver: does a given library subagent's model support vision?
            # Drives the macos-native read_screen text→image-combo swap. The
            # orchestrator's provider/model (computed above as
            # ``_orch_provider`` / ``_orch_model_id``) is the parent for
            # inherit subagents.
            def _subagent_vision(agent_spec: AgentSpec) -> bool:
                prov, mid = _provider_model_for_subagent(
                    agent_spec, config, _orch_provider, _orch_model_id,
                )
                try:
                    return supports_vision(prov, mid)
                except Exception:
                    return False

            subagents, claimed_ids = _build_subagents_from_library(
                mcp_mgr, standard_tools, model=orch_llm, backend=backend,
                ask_user_tools=ask_user_tools, pw_pool=pw_pool,
                memory_middleware=memory_middleware or None,
                activity_tools=activity_tools,
                subagent_vision_resolver=_subagent_vision,
                session_id=session_id,
            )
            from backend.mcp_manager import dedupe_tool_names

            connections = mcp_mgr.connections
            main_mcp_tools = dedupe_tool_names(
                (sid, conn.tools)
                for sid, conn in connections.items()
                if conn.connected and sid not in claimed_ids
            )
            # If no library subagent claimed macos-native, it stays on the main
            # orchestrator.  Apply the same vision swap when the orchestrator
            # model is vision-capable so the main agent also gets the text+image
            # read_screen and capture_app_screenshot.
            if "macos-native" not in claimed_ids and _orch_supports_vision:
                _macos_conn = connections.get("macos-native")
                if _macos_conn is not None and getattr(_macos_conn, "connected", False):
                    main_mcp_tools = _apply_macos_vision_variant(
                        main_mcp_tools, _macos_conn,
                    )
                    logger.info(
                        "macos-native: vision read_screen enabled for main orchestrator",
                    )
            extra_tools: list[Any] = []

            all_tools = (
                main_mcp_tools
                + standard_tools
                + management_tools
                + mcp_builder_tools
                + spawn_tools
                + session_search_tools
                + schedule_tools
                + trigger_tools
                + exo_tools
                + omlx_tools
                + settings_tools
                + privacy_tools
                + file_tools
                + activity_tools
                + recap_tools
                + ambient_tools
                + ask_user_tools
                + extra_tools
            )

            # Build the general-purpose subagent ourselves so we can:
            #  1. Wrap it in StreamingSubagentRunnable for live UI feedback
            #  2. Omit HumanInTheLoopMiddleware (subagents run inside the
            #     `task` tool — HITL interrupts have no channel back to
            #     the user and would cause the subagent to hang)
            gp_has_pw = pw_pool is not None and _PLAYWRIGHT_SERVER_ID not in claimed_ids
            gp_subagent = _build_gp_subagent(
                model=orch_llm, tools=all_tools, backend=backend,
                pw_pool=pw_pool if gp_has_pw else None,
                memory_middleware=memory_middleware or None,
                session_id=session_id,
            )
            if subagents is None:
                subagents = [gp_subagent]
            else:
                subagents.insert(0, gp_subagent)

            # The main orchestrator in GP mode holds playwright tools directly
            # only when no library sub-agent claimed them.
            main_has_playwright = gp_has_pw

        # Universal loop-guard chokepoint: guarantee every tool handed to the
        # orchestrator (and, by shared reference, the GP subagent) is wrapped.
        # Idempotent for the MCP/macOS/Playwright tools already guarded by
        # their per-connection loaders.
        _apply_universal_loop_guard(
            all_tools, scope="orchestrator", session_id=session_id
        )

        # Background memory extraction: write learnings to the agent's
        # persistent AGENTS.md (or a session-level file for general-purpose).
        # Scheduled runs write to a per-session file so unattended runs
        # don't pollute the global agent memory without review.
        extra_middleware: list[Any] = []

        # ReAct shim for text-only on-device models (certain MLX
        # templates).  Must sit at index 0 so it intercepts the model
        # request before any inner middleware tries to call bind_tools.
        # No-op when the model already supports native tool calling.
        react_shim = _maybe_react_shim(graph_llm)
        if react_shim is not None:
            extra_middleware.insert(0, react_shim)
            logger.info(
                "ReAct shim enabled for main orchestrator (model=%s)",
                type(graph_llm).__name__,
            )

        # The deepagents package includes AnthropicPromptCachingMiddleware in
        # its internal stack (harmless no-op for non-Anthropic models).  For
        # Bedrock we inject BedrockPromptCachingMiddleware here so the main
        # orchestrator agent also benefits from cached system prompts / tools.
        orchestrator_caching_mw = _make_prompt_caching_middleware(graph_llm)
        if orchestrator_caching_mw:
            extra_middleware.append(orchestrator_caching_mw)

        # Playwright snapshot pruning: the accessibility-tree YAML returned by
        # ``browser_snapshot`` / ``browser_take_screenshot`` (and the long
        # "### Snapshot" blocks appended to every browser tool result) balloon
        # the context window quickly.  Attach pruning whenever the main
        # orchestrator holds playwright tools directly — either because the
        # user selected a playwright-capable agent as the top-level agent, or
        # because no library sub-agent claimed ``playwright-mcp`` in GP mode.
        # Sub-agents that own playwright tools already get this middleware via
        # ``_build_subagents_from_library`` / ``_build_gp_subagent``.
        if main_has_playwright:
            from middleware.playwright_pruning import PlaywrightSnapshotPruningMiddleware
            extra_middleware.insert(0, PlaywrightSnapshotPruningMiddleware())
            logger.info(
                "Playwright snapshot pruning enabled for main orchestrator (agent=%s)",
                agent_name or "general-purpose",
            )

        # Runtime safety guards on the main orchestrator.  The
        # subagent-as-tool guard only matters here (the general-purpose
        # and library subagents don't dispatch to other library agents).
        from backend.safety_middleware import (
            ExecutePathSafetyMiddleware,
            HighRiskExecuteFlaggerMiddleware,
            LiveOutputMiddleware,
            SubagentAsToolGuardMiddleware,
        )
        extra_middleware.append(ExecutePathSafetyMiddleware())
        extra_middleware.append(HighRiskExecuteFlaggerMiddleware())
        extra_middleware.append(LiveOutputMiddleware(live_output_queue, files_dir))
        if subagents:
            extra_middleware.append(
                SubagentAsToolGuardMiddleware(
                    subagent_names=[sa["name"] for sa in subagents],
                )
            )

        if is_scheduled_run:
            memory_path = files_dir / "learnings.md"
        elif agent_name:
            from backend.agent_library import _agents_dir, _slugify
            memory_path = _agents_dir() / _slugify(agent_name) / "AGENTS.md"
        else:
            memory_path = files_dir / "learnings.md"
        extra_middleware.append(
            MemoryExtractionMiddleware(
                model=graph_llm,
                memory_path=memory_path,
                extract_every_n_turns=_MEMORY_EXTRACTION_INTERVAL,
            )
        )

        extra_middleware.extend(memory_middleware)

        # Relocate view_image tool-result images into a following user message
        # for OpenAI-compatible providers (omlx/exo/openai/azure), whose chat
        # templates drop image content from tool-role messages.  Appended
        # before context truncation so the relocated (most-recent) user image
        # message survives any clipping.  No-op for Anthropic/Bedrock/MLX.
        orchestrator_image_relocation = _maybe_tool_image_relocation(graph_llm)
        if orchestrator_image_relocation is not None:
            extra_middleware.append(orchestrator_image_relocation)
            logger.info(
                "Tool-image relocation enabled for main orchestrator (model=%s)",
                type(graph_llm).__name__,
            )

        # Fill in real filenames on read_file's PDF/PPTX attachments so
        # OpenAI-compatible servers (e.g. oMLX) don't reject them as an
        # unsupported attachment type. See middleware.file_attachment_filename.
        orchestrator_file_attachment_filename = _maybe_file_attachment_filename(graph_llm)
        if orchestrator_file_attachment_filename is not None:
            extra_middleware.append(orchestrator_file_attachment_filename)
            logger.info(
                "File-attachment filename fix enabled for main orchestrator (model=%s)",
                type(graph_llm).__name__,
            )

        # Last-resort context-window enforcement for the main orchestrator.
        # Appended AFTER every other middleware so it runs innermost — this
        # is the only middleware that can observe the fully-assembled prompt
        # produced by deepagents' hardcoded Todo / Filesystem / SubAgents
        # blocks plus all of our additions, and clip it down to the model's
        # actual budget.  No-op when the model has a large context window.
        orchestrator_ctx_trunc = _maybe_context_truncation(graph_llm)
        if orchestrator_ctx_trunc is not None:
            extra_middleware.append(orchestrator_ctx_trunc)
            logger.info(
                "Context truncation enabled for main orchestrator (model=%s, budget=%d tok)",
                type(graph_llm).__name__,
                orchestrator_ctx_trunc._budget,
            )

        if config.ambient_suggest_recurrence and not is_scheduled_run:
            system_prompt += (
                "\n\n## Ambient Scheduling\n"
                "After you finish a task that the user is likely to want repeated — "
                "for example: running a script, checking a website, monitoring a folder, "
                "generating a report, or any request containing words like 'every', 'daily', "
                "'weekly', 'monitor', 'remind', or 'watch' — ask a single, brief, natural "
                "follow-up question at the end of your response. Offer to automate it using "
                "one of these options:\n"
                "- **Schedule**: run it automatically on a recurring schedule "
                "(use the `create_schedule` tool with a cron expression, e.g. `0 9 * * *` for "
                "daily at 9am). Always suggest a sensible default cron based on what the user asked.\n"
                "- **Trigger**: run it automatically when an event occurs "
                "(use the `create_trigger` tool, e.g. when a new file appears in a folder).\n"
                "- **Repeat once**: run it again at a specific time "
                "(use `create_schedule` with a one-shot cron).\n"
                "Keep the question to one sentence. Only ask if the task genuinely looks "
                "repeatable — skip it for clearly one-off requests. "
                "If the user says yes, use the appropriate tool immediately without asking for more details, "
                "using sensible defaults. Do not ask about this more than once per conversation."
            )

        _now_utc = datetime.now(timezone.utc)
        _now_local = datetime.now()
        _tz_name = time.tzname[0]
        _agent_location = os.environ.get("AGENT_LOCATION", "")
        _location_line = f"\n- Location: {_agent_location}" if _agent_location else ""
        system_prompt += (
            f"\n\n## Session Context\n"
            f"- Save reports and output files under `/output/` (e.g. "
            f"`/output/report.md`) using the file tools. The file tools operate "
            f"in a virtual session filesystem; use virtual paths like `/output/...` "
            f"or `/input/...`, or a path relative to the session root.\n"
            f"- The real session files directory is `{files_dir}` (also available "
            f"as the `SESSION_FILES` env var). Use this absolute path only with the "
            f"`execute`/shell tool; the file tools also accept it but virtual paths "
            f"are preferred.\n"
            f"- When a script run via `execute` (e.g. a Python heredoc) reads or "
            f"writes files, never hardcode `/output/...` (it resolves to the host "
            f"root) and never put the literal string `$SESSION_FILES` inside the "
            f"script source (it is not expanded there). Inside scripts use "
            f"`os.environ['SESSION_FILES']` or a path relative to the working "
            f"directory (e.g. `open('output/file.html')`), then confirm with "
            f"`read_file('/output/file.html')`.\n"
            f"- To attach or upload a local file to a web page, do NOT read the "
            f"file's contents. Click the page's upload control to open the file "
            f"chooser, then call `browser_file_upload` with the file's REAL "
            f"absolute path (e.g. `{files_dir}/uploads/<name>`), not a virtual "
            f"`/uploads/...` path — the browser tools run in a separate process "
            f"and only understand real filesystem paths. Reading binary files "
            f"(PDF, DOCX, images) into the conversation is unnecessary for "
            f"uploading and may fail.\n"
            f"- Current date/time: {_now_local.strftime('%A, %B %d, %Y %H:%M')} {_tz_name}"
            f" ({_now_utc.strftime('%H:%M UTC')})"
            f"{_location_line}\n"
        )

        system_prompt += (
            "\n## Data Integrity (NON-NEGOTIABLE)\n"
            "- Report ONLY data that tools or subagents actually returned. NEVER "
            "invent, guess, or extrapolate URLs, citations, IDs, counts, prices, or "
            "any other value, and NEVER emit placeholder or \"example\" URLs.\n"
            "- When you compile a report from subagent results, every row/link must "
            "trace back to a real value a subagent returned or a file it wrote. If a "
            "subagent did not return the data you need, treat it as MISSING: say so "
            "explicitly (and re-run the subagent if it matters) — do not fabricate to "
            "fill the gap.\n"
            "- Before citing aggregate totals, verify them against the actual source "
            "files/summaries; do not state counts you cannot back with returned data.\n"
            "\n## Parallel Subagents & Output Files\n"
            "- The session filesystem (including `/output/`) is SHARED by every "
            "subagent. If you spawn multiple subagents in parallel that each write a "
            "result/output file, give each one a DISTINCT output path in its task "
            "description (e.g. `/output/result-melbourne.json`, "
            "`/output/result-france.json`). Two parallel subagents must never write the "
            "same file — the later writer silently overwrites the earlier one.\n"
            "- Tell each subagent the exact output path to use, and after it finishes "
            "read back that specific file (or its returned summary) rather than assuming "
            "a single shared `result.json` holds everyone's data.\n"
        )

        if is_scheduled_run and schedule_id:
            system_prompt += (
                f"- This session was initiated automatically by schedule "
                f"`{schedule_id}`. Use this schedule ID with the schedule tools "
                f"(e.g. `get_schedule_runs`) to inspect prior runs of this same "
                f"schedule when continuity with previous runs is useful.\n"
            )

        _insert_repeated_thought_guard(extra_middleware, scope="orchestrator")
        graph = await asyncio.to_thread(
            create_deep_agent,
            model=graph_llm,
            system_prompt=system_prompt,
            tools=all_tools,
            subagents=subagents,
            checkpointer=checkpointer,
            backend=backend,
            interrupt_on={"execute": True},
            middleware=extra_middleware,
        )

        logger.info(
            "[build_graph] DONE session=%s in %.1fs (%d tools, %d subagents)",
            session_id, time.monotonic() - t_graph_start,
            len(all_tools), len(subagents) if subagents else 0,
        )
        return graph, mcp_mgr, live_output_queue

    async def create_session(
        self,
        config: AppConfig,
        agent_name: Optional[str] = None,
        *,
        is_scheduled_run: bool = False,
        schedule_id: Optional[str] = None,
        trigger_source: Optional[str] = None,
        trigger_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        chain_depth: int = 0,
    ) -> Session:
        config.apply_to_environ()
        session_id = str(uuid.uuid4())

        sqlite_conn = await _connect_checkpoint_db()
        checkpointer = AsyncSqliteSaver(sqlite_conn)
        await checkpointer.setup()

        graph, mcp_mgr, live_output_queue = await self._build_graph(
            config, agent_name, session_id, checkpointer,
            is_scheduled_run=is_scheduled_run,
            schedule_id=schedule_id,
        )

        session = Session(
            session_id=session_id,
            agent_name=agent_name,
            graph=graph,
            tool_set=mcp_mgr,
            checkpointer=checkpointer,
            _sqlite_conn=sqlite_conn,
            schedule_id=schedule_id,
            trigger_source=trigger_source,
            trigger_id=trigger_id,
            parent_session_id=parent_session_id,
            chain_depth=chain_depth,
            llm_provider=config.llm.provider,
        )
        session.live_output_queue = live_output_queue
        # Drives the "Searching memory…" UI event per turn; only the realtime
        # layer actually performs per-turn retrieval.
        session.memory_inject = config.memory.effective_inject_realtime
        session.recursion_limit = max(1, min(config.orchestrator.recursion_limit, 10000))
        self._active[session_id] = session
        await session.save_meta_async()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._active.get(session_id)

    # Hard cap on parent → child → grandchild chains.  When the orchestrator
    # in a child session itself calls ``spawn_followup_session`` the chain
    # depth grows by one; once it hits this cap further spawns are rejected
    # with an error string the agent sees as a tool result.  Two levels of
    # auto-handoff is plenty for "build tools then use them" workflows; more
    # than that almost always indicates a confused agent in a loop.
    MAX_SESSION_CHAIN_DEPTH: int = 2

    async def spawn_child_session(
        self,
        parent_session_id: str,
        prompt: str,
    ) -> Session:
        """Create a fresh session linked to *parent_session_id* and inheriting
        its agent selection.

        The child session has its own LangGraph (rebuilt against the current
        ``AppConfig``, so any MCP servers / sub-agents created during the
        parent's turn are now bound to it) and its own checkpoint thread.
        The parent's session-files directory is **shared**, so files the
        parent wrote during the build phase remain visible to the child.

        The caller is responsible for actually firing *prompt* on the new
        session; this method only sets up state.  See
        :func:`backend.session_dispatch.kick_off_message` for the matching
        helper that schedules the agent task.

        Raises:
            ValueError: If the parent doesn't exist, or the spawn chain
                        would exceed :attr:`MAX_SESSION_CHAIN_DEPTH`.
        """
        parent = await self._ensure_session(parent_session_id)
        if parent is None:
            raise ValueError(f"Parent session {parent_session_id!r} not found")

        if parent.chain_depth >= self.MAX_SESSION_CHAIN_DEPTH:
            raise ValueError(
                f"Spawn chain depth {parent.chain_depth} already at cap "
                f"({self.MAX_SESSION_CHAIN_DEPTH}). Refusing to spawn another "
                f"child to avoid runaway loops."
            )

        cfg = await AppConfig.aload()
        child = await self.create_session(
            config=cfg,
            agent_name=parent.agent_name,
            parent_session_id=parent_session_id,
            chain_depth=parent.chain_depth + 1,
            trigger_source="spawn",
        )
        # Pre-fill the title so the session list is informative before the
        # first turn finishes streaming.
        child.title = (prompt[:60] + "…") if len(prompt) > 60 else prompt
        await child.save_meta_async()
        logger.info(
            "Spawned child session %s (depth=%d) from parent %s",
            child.id, child.chain_depth, parent_session_id,
        )
        return child

    async def resume_session(
        self,
        session_id: str,
        config: AppConfig,
    ) -> Optional[Session]:
        """Resume a previously saved session by rebuilding the agent with the persisted checkpoint."""
        if session_id in self._active:
            return self._active[session_id]

        meta_path = _sessions_dir() / f"{session_id}.json"
        if not await asyncio.to_thread(meta_path.exists):
            return None

        try:
            raw = await asyncio.to_thread(meta_path.read_text, "utf-8")
            meta = json.loads(raw)
        except Exception:
            logger.warning("Failed to parse session meta for %s", session_id, exc_info=True)
            return None

        info = SessionInfo.model_validate(meta)
        config.apply_to_environ()

        sqlite_conn = await _connect_checkpoint_db()
        checkpointer = AsyncSqliteSaver(sqlite_conn)
        await checkpointer.setup()

        graph, mcp_mgr, live_output_queue = await self._build_graph(
            config, info.agent_name, session_id, checkpointer,
            is_scheduled_run=info.trigger_source == "schedule",
            schedule_id=info.schedule_id,
        )

        session = Session(
            session_id=session_id,
            agent_name=info.agent_name,
            graph=graph,
            tool_set=mcp_mgr,
            checkpointer=checkpointer,
            _sqlite_conn=sqlite_conn,
            schedule_id=info.schedule_id,
            trigger_source=info.trigger_source,
            trigger_id=info.trigger_id,
            parent_session_id=info.parent_session_id,
            chain_depth=info.chain_depth,
            llm_provider=config.llm.provider,
        )
        session.live_output_queue = live_output_queue
        session.memory_inject = config.memory.effective_inject_realtime
        session.title = info.title
        session.message_count = info.message_count
        session.tools_used = info.tools_used
        session.created_at = info.created_at
        session.updated_at = info.updated_at
        # Restore persisted run metrics. A session only reaches resume_session
        # when it is *not* already active in memory, so a persisted "running"
        # status is necessarily stale (the run was interrupted by a crash or a
        # terminal path that failed to stamp a final status). Downgrade it to
        # "error" so the UI doesn't resurrect a ghost "running" entry.
        if info.status == "running":
            session.status = "error"
            session.error = info.error or "Interrupted — run did not finish"
            session.error_code = info.error_code
            session.finished_at = info.finished_at or info.updated_at
        else:
            session.status = info.status
            session.finished_at = info.finished_at
            session.error = info.error
            session.error_code = info.error_code
        session.duration_ms = info.duration_ms
        session.model = info.model
        session.input_tokens = info.input_tokens or 0
        session.output_tokens = info.output_tokens or 0
        session.estimated_cost_usd = info.estimated_cost_usd or 0.0
        # Restore end-of-run evaluation summary so the in-memory session (which
        # shadows the on-disk meta in /api/runs via list_active) doesn't reset
        # these to None and hide a completed evaluation's score in the UI.
        session.eval_status = info.eval_status
        session.eval_overall_score = info.eval_overall_score
        session.eval_pass_count = info.eval_pass_count
        session.eval_total = info.eval_total
        self._active[session_id] = session
        logger.info("Resumed session %s (%s)", session_id, info.title)
        return session

    async def refresh_tools(self, config: AppConfig) -> None:
        """Rebuild the agent graph for all active sessions with fresh tools from config.

        Called when excluded_tools changes so active sessions pick up the update.
        If the rebuild fails (e.g. MLX model not yet downloaded) the session
        keeps its existing graph rather than being torn down.
        """
        config.apply_to_environ()

        for session in list(self._active.values()):
            old_tool_set = session.tool_set
            try:
                graph, mcp_mgr, _ = await self._build_graph(
                    config, session.agent_name, session.id, session._checkpointer,
                    is_scheduled_run=session.trigger_source == "schedule",
                    schedule_id=session.schedule_id,
                    live_output_queue=session.live_output_queue,
                )
            except Exception as exc:
                logger.warning(
                    "refresh_tools: _build_graph failed for session %s — "
                    "keeping existing graph. Reason: %s",
                    session.id, exc,
                )
                continue
            session.graph = graph
            session.tool_set = mcp_mgr
            if old_tool_set is not None:
                try:
                    await old_tool_set.close()
                except Exception:
                    logger.debug("Error closing old tool set for session %s", session.id, exc_info=True)
            logger.info("Refreshed tools for session %s", session.id)

        # Swapping out the old graphs above drops the only remaining strong
        # references to any MLX chat model that the *previous* provider held
        # (the process-wide weight cache is cleared separately by the
        # /api/mlx/unload handler the UI calls when switching away).  Force a
        # GC + Metal allocator reclaim here so the freed weight buffers are
        # returned to the OS rather than lingering in MLX's free pool until
        # the next allocation.  ``mx.clear_cache`` only releases the unused
        # pool, so this is safe even when MLX is still the active provider.
        await asyncio.to_thread(_reclaim_mlx_memory)

    async def close_session(self, session_id: str) -> None:
        session = self._active.pop(session_id, None)
        if session:
            await session.save_meta_async()
            await session.close()

    async def delete_session(self, session_id: str) -> None:
        """Close an active session (if any) and remove its files from disk."""
        import shutil

        from backend.session_transcript import delete_transcript

        session = self._active.pop(session_id, None)
        if session:
            await session.close()

        def _remove_session_files() -> None:
            meta_path = _sessions_dir() / f"{session_id}.json"
            msgs_path = _messages_path(session_id)
            eval_path = _sessions_dir() / f"{session_id}.eval.json"
            session_dir = _sessions_dir() / session_id
            meta_path.unlink(missing_ok=True)
            msgs_path.unlink(missing_ok=True)
            eval_path.unlink(missing_ok=True)
            delete_transcript(session_id)
            if session_dir.exists():
                shutil.rmtree(session_dir, ignore_errors=True)

        await asyncio.to_thread(_remove_session_files)
        try:
            db = await _connect_checkpoint_db()
            try:
                await db.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
                await db.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
                await db.commit()
                # Rows deleted above only land in SQLite's internal freelist —
                # the file itself never shrinks without this. incremental_vacuum
                # only does anything once auto_vacuum=INCREMENTAL has taken
                # effect (see _connect_checkpoint_db), but is a harmless no-op
                # otherwise, so it's always safe to call here.
                await db.execute("PRAGMA incremental_vacuum")
            finally:
                await db.close()
        except Exception:
            logger.debug("Error cleaning checkpoint DB for session %s", session_id, exc_info=True)

    async def evict_idle(self, busy_session_ids: set[str] | None = None) -> int:
        """Close sessions idle longer than ``_SESSION_IDLE_TIMEOUT_SECS``.

        Sessions whose IDs are in *busy_session_ids* (e.g. those with a
        running agent task) are skipped.  Returns the number of sessions
        evicted.
        """
        now = datetime.now(timezone.utc)
        busy = busy_session_ids or set()
        to_evict: list[str] = []

        for sid, session in self._active.items():
            if sid in busy:
                continue
            idle_secs = (now - session.updated_at).total_seconds()
            if idle_secs > _SESSION_IDLE_TIMEOUT_SECS:
                to_evict.append(sid)

        for sid in to_evict:
            logger.info("Evicting idle session %s (idle > %ds)", sid, _SESSION_IDLE_TIMEOUT_SECS)
            await self.close_session(sid)

        return len(to_evict)

    async def close_all(self) -> None:
        if self._eviction_task is not None:
            self._eviction_task.cancel()
            self._eviction_task = None
        for sid in list(self._active.keys()):
            await self.close_session(sid)

    async def compact_checkpoint_db(self) -> None:
        """One-time migration + ongoing maintenance for checkpoints.sqlite.

        Historically ``delete_session`` deleted rows without ever running
        ``VACUUM``/``incremental_vacuum``, so on ``auto_vacuum=NONE`` (SQLite's
        default) freed pages stayed in the file's internal freelist forever —
        the file only ever grew, even though live row counts stayed small.

        Runs in a thread (blocking SQLite calls) and is best-effort: skipped
        entirely if another connection currently holds the write lock (e.g.
        an active session mid-checkpoint), since this is opportunistic
        housekeeping, not correctness-critical.
        """
        await asyncio.to_thread(self._compact_checkpoint_db_sync)

    @staticmethod
    def _compact_checkpoint_db_sync() -> None:
        import sqlite3

        db_path = Path(_checkpoint_db_path())
        if not db_path.exists():
            return
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
        except sqlite3.OperationalError:
            return
        try:
            page_count = conn.execute("PRAGMA page_count").fetchone()[0]
            if not page_count:
                return
            auto_vacuum = conn.execute("PRAGMA auto_vacuum").fetchone()[0]
            freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
            free_ratio = freelist_count / page_count
            if auto_vacuum == 0 and free_ratio > 0.1:
                logger.info(
                    "checkpoints.sqlite: auto_vacuum=NONE with %.0f%% free pages "
                    "(%d/%d) — running one-time VACUUM to reclaim disk space and "
                    "switch to incremental auto-vacuum",
                    free_ratio * 100, freelist_count, page_count,
                )
                conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
                conn.execute("VACUUM")
            elif freelist_count:
                conn.execute("PRAGMA incremental_vacuum")
        except sqlite3.OperationalError:
            logger.debug("checkpoints.sqlite compaction skipped (locked)", exc_info=True)
        finally:
            conn.close()

    def list_active(self) -> list[SessionInfo]:
        return [s.to_info() for s in self._active.values()]

    def list_history(self) -> list[SessionInfo]:
        sessions: list[SessionInfo] = []
        for p in sorted(_sessions_dir().glob("*.json"), reverse=True):
            if p.name.endswith((".messages.json", ".eval.json")):
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                sessions.append(SessionInfo.model_validate(data))
            except Exception:
                logger.debug("Skipping corrupt session meta: %s", p.name, exc_info=True)
                continue
        return sessions[:_MAX_SESSION_HISTORY]

    async def _ensure_session(self, session_id: str) -> Optional[Session]:
        session = self._active.get(session_id)
        if not session:
            session = await self.resume_session(session_id, await AppConfig.aload())
        if session:
            session.updated_at = datetime.now(timezone.utc)
        return session

    async def _generate_title(
        self,
        session: Session,
        session_id: str,
        user_query: str,
    ) -> None:
        """Use a lightweight LLM call to produce a short descriptive title."""
        try:
            from deep_agent.model_factory import create_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            # create_llm can block (MLX/oMLX may load model weights
            # synchronously) — keep it off the event loop.
            provider = (await AppConfig.aload()).llm.provider
            llm = await asyncio.to_thread(create_llm, provider)
            messages = [
                SystemMessage(
                    content=(
                        "Generate a concise title (max 6 words) that captures "
                        "the intent of the user's request. Return ONLY the "
                        "title text, nothing else."
                    ),
                ),
                HumanMessage(content=user_query),
            ]
            resp = await llm.ainvoke(messages)
            title = (resp.content if isinstance(resp.content, str) else str(resp.content)).strip().strip('"')
            if title:
                session.title = title[:80]
                await session.save_meta_async()
                logger.debug("Generated title for session %s: %s", session_id, session.title)
        except Exception:
            logger.debug("Title generation failed for session %s — keeping default", session_id, exc_info=True)

    # ------------------------------------------------------------------
    # Memory gate — background memory consolidation
    # ------------------------------------------------------------------

    async def _maybe_consolidate_memory(self, session: Session) -> None:
        """Evaluate whether to kick off a background memory consolidation.

        Best-effort — failures are logged and swallowed so they never
        disrupt the user's stream response.
        """
        try:
            if session.schedule_id:
                return

            cfg = await AppConfig.aload()
            mcfg = cfg.memory
            if not mcfg.enabled:
                return

            from backend.consolidation_lock import last_consolidated_at, try_acquire
            from backend.session_transcript import list_transcripts_since

            since = last_consolidated_at()
            now_ms = time.time() * 1000
            interval_ms = mcfg.min_hours * 3600 * 1000
            if (now_ms - since) < interval_ms:
                return

            candidates = list_transcripts_since(since)
            if len(candidates) < mcfg.min_sessions:
                return

            prev_mtime = try_acquire()
            if prev_mtime is None:
                return

            logger.info(
                "[memory] gate passed — %d transcript(s), launching consolidation",
                len(candidates),
            )

            from backend.memory import execute_consolidation
            asyncio.create_task(execute_consolidation(prev_mtime, candidates, mcfg))
        except Exception:
            logger.debug("Memory gate check failed", exc_info=True)

    async def _maybe_index_transcript(self, session: Session) -> None:
        """Background-index the just-completed session transcript.

        Runs only when memory.embedding_enabled is True.  Best-effort —
        any failure is logged and swallowed so it never affects the stream.
        """
        try:
            cfg = await AppConfig.aload()
            if not cfg.memory.embedding_enabled:
                return
            from backend.embedding_index import get_embedding_index
            idx = await get_embedding_index()
            asyncio.create_task(idx.index_transcript(session.session_id))
        except Exception:
            logger.debug("Transcript index trigger failed", exc_info=True)

    async def _maybe_ambient_sweep(self, session: Session) -> None:
        """Trigger a debounced ambient sweep after a session completes.

        Skipped for sessions that were themselves spawned by the ambient
        assistant (to avoid feedback loops) or scheduled runs.  Best-effort.
        """
        try:
            cfg = await AppConfig.aload()
            if not cfg.ambient.enabled:
                return
            if not cfg.ambient.react_to_session_end:
                return
            # Don't sweep after ambient-spawned or scheduled sessions.
            if session.trigger_source in ("ambient", "schedule"):
                return

            from backend.ambient_agent import run_sweep
            # Small delay so memory consolidation / transcript indexing finish
            # first and are available as context for the sweep.
            async def _delayed() -> None:
                await asyncio.sleep(15)
                await run_sweep(is_manual=False)

            asyncio.create_task(_delayed())
        except Exception:
            logger.debug("Ambient sweep trigger failed", exc_info=True)

    async def _do_stream(
        self,
        session: Session,
        session_id: str,
        astream_iter: Any,
        printed_offset: int = 0,
        context_queue: Optional[Any] = None,
        run_config: Optional[dict] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        from langchain_core.messages import AIMessage, ToolMessage

        printed = printed_offset
        aiter = astream_iter.__aiter__()
        _pending_tools: dict[str, tuple[str, float]] = {}
        _chunk_idx = 0
        t_stream_start = time.monotonic()
        live_q: Optional[asyncio.Queue] = getattr(session, "live_output_queue", None)
        logger.info("[stream] START session=%s", session_id)
        while True:
            try:
                t_wait = time.monotonic()
                chunk = await aiter.__anext__()
                wait_ms = (time.monotonic() - t_wait) * 1000
                _chunk_idx += 1
                if wait_ms > 2000:
                    logger.info("[stream] chunk #%d arrived after %.0fms wait", _chunk_idx, wait_ms)
            except StopAsyncIteration:
                # Flush any remaining live-output lines before signalling done.
                if live_q is not None:
                    while not live_q.empty():
                        yield live_q.get_nowait()
                logger.info("[stream] DONE session=%s total=%.1fs chunks=%d", session_id, time.monotonic() - t_stream_start, _chunk_idx)
                break
            # Drain live-output lines that arrived while awaiting this chunk.
            # The queue is populated by LiveOutputMiddleware's tee polling task;
            # since the tee runs during tool execution (between the tool_call
            # and tool_result chunks), draining here ensures execute_output
            # events reach the frontend before the tool_result message.
            if live_q is not None:
                while not live_q.empty():
                    yield live_q.get_nowait()
            if "messages" not in chunk:
                continue
            new_msgs = chunk["messages"][printed:]
            printed = len(chunk["messages"])

            for msg in new_msgs:
                if isinstance(msg, AIMessage):
                    memory_topics = msg.response_metadata.get("memory_topics")
                    if memory_topics:
                        yield {"type": "memory_context", "content": "", "metadata": {"topics": memory_topics}}

                    content = extract_text_content(msg.content)

                    # Extract MLX generation stats (cache hit ratio, TPS, peak
                    # memory, etc.) built by ChatMLXText.  Done once per
                    # AIMessage so both tool-call and text turns contribute to
                    # the session-level throughput accumulators and so the
                    # frontend can render a live stats panel even during long
                    # tool-calling sequences.
                    rmeta = msg.response_metadata or {}
                    stats = {
                        k: rmeta[k] for k in (
                            "tokens_from_cache",
                            "tokens_prefilled",
                            "cache_hit_ratio",
                            "prompt_tps",
                            "generation_tokens",
                            "generation_tps",
                            "peak_memory_gb",
                        ) if k in rmeta
                    }
                    logger.debug(
                        "[stream] AIMessage response_metadata keys=%s stats_found=%s",
                        list(rmeta.keys())[:10] if rmeta else [],
                        list(stats.keys()) if stats else "none",
                    )
                    session.accumulate_mlx_stats(stats)
                    # Accumulate token usage for frontier providers
                    # (Anthropic / OpenAI response_metadata) and MLX.
                    usage = getattr(msg, "usage_metadata", None) or {}
                    if not usage:
                        # LangChain < 0.3 stores usage in response_metadata
                        usage = rmeta.get("token_usage") or rmeta.get("usage") or {}
                    in_tok = int(
                        usage.get("input_tokens") or usage.get("prompt_tokens") or 0
                    )
                    # MLX doesn't populate usage_metadata; fall back to the
                    # total prompt tokens visible this turn (freshly encoded
                    # tokens_prefilled + tokens served from the KV prefix
                    # cache).  Without this, input_tokens stays 0 for MLX
                    # sessions and the dashboard TIPS aggregate is skipped
                    # because the weighting guard ``if ptps and in_tok``
                    # never fires.
                    if not in_tok and stats:
                        in_tok = int(
                            (stats.get("tokens_prefilled") or 0)
                            + (stats.get("tokens_from_cache") or 0)
                        )
                    out_tok = int(
                        usage.get("output_tokens") or usage.get("completion_tokens")
                        or stats.get("generation_tokens") or 0
                    )
                    if in_tok or out_tok:
                        session.input_tokens += in_tok
                        session.output_tokens += out_tok

                    if msg.tool_calls:
                        # Emit tool_call rows; skip any text in the same
                        # AIMessage.  Models sometimes include a preamble or
                        # summary ("I'll run these…") alongside tool_calls.
                        # Suppressing it here keeps the approval bubble
                        # directly below the tool rows without stray text
                        # appearing above or between them.  The final summary
                        # always arrives in a text-only AIMessage after the
                        # tool results, so nothing meaningful is lost.
                        for idx, tc in enumerate(msg.tool_calls):
                            tool_name = tc.get("name", "")
                            tc_id = tc.get("id", "")
                            logger.info("[stream] TOOL_CALL %s (id=%s)", tool_name, tc_id)
                            _pending_tools[tc_id] = (tool_name, time.monotonic())
                            if tool_name and tool_name not in session.tools_used:
                                session.tools_used.append(tool_name)
                            tc_meta: dict[str, Any] = {"args": tc.get("args", {})}
                            if tc_id:
                                tc_meta["tool_call_id"] = tc_id
                            # Attach the turn's stats to the first tool_call row
                            # only so the live session panel counts each turn
                            # once (a single AIMessage may emit several calls).
                            if stats and idx == 0:
                                tc_meta["stats"] = stats
                            resp = {
                                "type": "tool_call",
                                "content": tool_name,
                                "metadata": tc_meta,
                            }
                            await _append_message_async(session_id, resp)
                            await _transcript(
                                session_id, "tool_call", tc.get("args", {}),
                                tool_name=tool_name, tool_call_id=tc_id,
                            )
                            yield resp
                    elif content:
                        resp = {"type": "agent", "content": content}
                        # Forward the model's hidden reasoning (set by the
                        # MLX ReAct wrapper / middleware via
                        # ``additional_kwargs["thought"]``) so the frontend
                        # can render it as a collapsible Thinking block.
                        thought = (msg.additional_kwargs or {}).get("thought")
                        meta: dict[str, Any] = {}
                        if memory_topics:
                            meta["memory_topics"] = memory_topics
                        if thought:
                            meta["thought"] = thought
                        if stats:
                            meta["stats"] = stats
                        if meta:
                            resp["metadata"] = meta
                        await _append_message_async(session_id, resp)
                        await _transcript(session_id, "assistant", content, role="assistant")
                        yield resp
                elif isinstance(msg, ToolMessage):
                    metadata: dict[str, Any] = {"name": getattr(msg, "name", "tool")}
                    tc_id_done = getattr(msg, "tool_call_id", None)
                    if tc_id_done:
                        metadata["tool_call_id"] = tc_id_done
                        started = _pending_tools.pop(tc_id_done, None)
                        if started:
                            logger.info("[stream] TOOL_RESULT %s (id=%s) took %.1fs", started[0], tc_id_done, time.monotonic() - started[1])
                        else:
                            logger.info("[stream] TOOL_RESULT %s (id=%s)", metadata.get("name"), tc_id_done)

                    # Full content for transcript (dream mode needs untruncated output)
                    if isinstance(msg.content, list):
                        full_text = " ".join(
                            block.get("text", "")
                            for block in msg.content
                            if isinstance(block, dict) and block.get("type") == "text"
                        )
                    else:
                        full_text = str(msg.content)
                    await _transcript(
                        session_id, "tool_result", full_text,
                        tool_name=metadata.get("name"),
                        tool_call_id=tc_id_done,
                    )

                    # UI message history keeps a (possibly truncated) preview.
                    # For execute results we show head + tail lines so the UI
                    # can render both the start and end of long output.
                    images: list[dict[str, str]] = []
                    if isinstance(msg.content, list):
                        text_parts = []
                        for block in msg.content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif block.get("type") == "image" and block.get("base64"):
                                    images.append({
                                        "base64": block["base64"],
                                        "mime_type": block.get("mime_type", "image/png"),
                                    })
                                # OpenAI-style image_url block with a base64 data
                                # URL — what the macos vision tools (read_screen
                                # combo / capture_app_screenshot) emit. Mirrors
                                # the extraction in streaming_subagent.py.
                                elif block.get("type") == "image_url":
                                    url = (block.get("image_url") or {}).get("url", "")
                                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                                        header, b64 = url.split(";base64,", 1)
                                        mime = header[len("data:"):] or "image/png"
                                        images.append({"base64": b64, "mime_type": mime})
                        combined = " ".join(text_parts)
                    else:
                        combined = str(msg.content)

                    tool_name_done = metadata.get("name", "")
                    if tool_name_done == "execute":
                        preview, total_lines, truncated = _head_tail_preview(combined)
                        metadata["output_lines"] = total_lines
                        metadata["output_truncated"] = truncated
                    else:
                        preview = combined[:500]

                    if images:
                        metadata["images"] = images
                    persisted = {
                        "type": "tool_result",
                        "content": preview,
                        "metadata": metadata,
                    }
                    await _append_message_async(session_id, persisted)
                    yield persisted

            # --- Level 2 context injection (safe boundaries only) ---
            # We may only inject a HumanMessage between a completed "tools"
            # node and the upcoming "model" node.  Injecting after the "model"
            # node (when the last message is an AIMessage with pending
            # tool_calls) would insert HumanMessage between tool_use and
            # tool_result blocks — Anthropic rejects that with a 400 error.
            #
            # Safe to inject when:
            #   • last new message is a ToolMessage  → tools node just finished
            #   • last new message is AIMessage with no tool_calls  → model gave
            #     a pure-text reply (no tools pending), safe to interject before
            #     the run ends naturally
            if context_queue is not None and run_config is not None and new_msgs:
                last_new = new_msgs[-1]
                is_safe_boundary = isinstance(last_new, ToolMessage) or (
                    isinstance(last_new, AIMessage) and not last_new.tool_calls
                )
                if is_safe_boundary:
                    try:
                        ctx = context_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        ctx = None
                    if ctx:
                        from langchain_core.messages import HumanMessage as _HM
                        await session.graph.aupdate_state(
                            run_config,
                            {"messages": [_HM(content=ctx)]},
                        )
                        logger.info("[stream] context injected mid-run session=%s", session_id)
                        # Signal the caller to restart astream from the updated checkpoint.
                        # Include `printed` so the next _do_stream call skips already-emitted messages.
                        yield {"type": "_context_injected", "content": ctx, "metadata": {"printed": printed}}
                        break

    @staticmethod
    def _interrupt_value_to_message(hitl_value: Any) -> dict[str, Any]:
        """Translate a LangGraph interrupt payload into the WS message dict
        the frontend understands.  Pure function — no I/O."""
        if isinstance(hitl_value, dict) and hitl_value.get("type") == "ask_user":
            return {
                "type": "ask_user",
                "content": hitl_value["question"],
                "metadata": {
                    k: v
                    for k, v in hitl_value.items()
                    if k not in ("type", "question")
                },
            }
        return {
            "type": "hitl_request",
            "content": "Tool execution requires approval",
            "metadata": hitl_value,
        }

    async def get_pending_interrupt(self, session_id: str) -> Optional[dict[str, Any]]:
        """Return the WS message for a paused interrupt, or None.

        Read-only: does not mutate graph state, does not persist anything.
        Used by the WS handler to re-emit pending interrupts on (re)connect
        so a dropped delivery (backend restart, mid-send WS close) doesn't
        leave the client unaware that the agent is waiting for them.
        """
        session = await self._ensure_session(session_id)
        if not session:
            return None

        run_config = {"configurable": {"thread_id": session_id}}
        state = await session.graph.aget_state(run_config)

        for task in (state.tasks or ()):
            for intr in (getattr(task, "interrupts", None) or ()):
                return self._interrupt_value_to_message(intr.value)
        return None

    async def _check_interrupts_or_done(
        self,
        session: Session,
        session_id: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """After a stream ends, check for pending HITL interrupts.

        Yields either a ``hitl_request`` (graph is paused) or ``done``.
        """
        run_config = {"configurable": {"thread_id": session_id}}
        state = await session.graph.aget_state(run_config)

        pending = []
        for task in (state.tasks or ()):
            for intr in (getattr(task, "interrupts", None) or ()):
                pending.append(intr)

        if pending:
            resp = self._interrupt_value_to_message(pending[0].value)
            await _append_message_async(session_id, resp)
            # Unattended auto-runs (scheduler / triggers) have no human present:
            # the driver immediately auto-approves the interrupt and resumes, so
            # the run is effectively still executing. Persisting "awaiting_input"
            # for these would misreport them as blocked on the Runs page. Only
            # attended sessions (e.g. a manual chat-page session) genuinely wait
            # on the user, so keep "awaiting_input" for those.
            if session.trigger_source in ("schedule", "trigger"):
                session.status = "running"
            else:
                session.status = "awaiting_input"
            yield resp
            await session.save_meta_async()
            return

        # Mark run complete and compute cost
        now = datetime.now(timezone.utc)
        session.status = "completed"
        session.finished_at = now
        session.duration_ms = int((now - session.created_at).total_seconds() * 1000)
        try:
            from backend.config import estimate_cost_usd
            cfg = await AppConfig.aload()
            # ``getattr(cfg.llm, provider)`` returns the provider *config
            # object* (e.g. AnthropicConfig), not a string — extract its
            # ``model_name`` so ``session.model`` stays a str (SessionInfo
            # validates it as one).
            model = session.model
            if not model:
                provider_cfg = getattr(cfg.llm, cfg.llm.provider, None)
                model = getattr(provider_cfg, "model_name", "") if provider_cfg else ""
            session.model = model or session.model
            session.estimated_cost_usd = estimate_cost_usd(
                model or "", session.input_tokens, session.output_tokens
            )
        except Exception:
            logger.debug("Cost estimation failed", exc_info=True)
        # Finalize *before* yielding ``done``: consumers (notably the scheduler's
        # drain loop) return/break the moment they see ``done``, which closes this
        # generator and skips any code after the yield. Persisting meta and
        # kicking off the (fire-and-forget) evaluation here guarantees both run
        # for every completion path, not just ones that keep iterating.
        await session.save_meta_async()
        await self._maybe_evaluate_run(session)
        yield {"type": "done", "content": ""}

    async def _maybe_evaluate_run(self, session: "Session") -> None:
        """Kick off a background end-of-run evaluation when enabled.

        Fire-and-forget (same pattern as memory consolidation / transcript
        indexing) so the run's ``done`` event is never delayed.  Best-effort.
        """
        try:
            cfg = await AppConfig.aload()
            if not cfg.evaluation.auto_evaluate:
                return
            from backend.eval_runner import launch_evaluation
            launch_evaluation(session.id)
        except Exception:
            logger.debug("Auto-evaluation trigger failed", exc_info=True)

    async def maybe_analyze_error(self, session_id: str) -> None:
        """Kick off background error analysis for a failed run when enabled.

        Mirrors :meth:`_maybe_evaluate_run` but targets runs that ended in
        ``error``: the analyzer classifies the failure and, only for
        prompt-addressable errors, drafts a stronger prompt. Fire-and-forget
        and best-effort so the error path is never delayed or broken. The
        analyzer reads the session's persisted meta + transcript from disk, so
        callers must have saved the error status before invoking this.
        """
        try:
            cfg = await AppConfig.aload()
            if not cfg.evaluation.analyze_errors:
                return
            from backend.eval_runner import launch_error_analysis
            launch_error_analysis(session_id)
        except Exception:
            logger.debug("Error-analysis trigger failed", exc_info=True)

    @staticmethod
    async def _check_privacy_lock(session: "Session") -> dict[str, Any] | None:
        """Return an error dict if the privacy lock is engaged and the
        session's LLM provider is a cloud provider, else None.

        This guard runs on *every turn* so that engaging the lock
        mid-session immediately blocks the next message — the LLM object
        was already constructed at session start, so the guard in
        ``create_llm`` would not fire again without this check.
        """
        try:
            from backend.privacy_lock import PrivacyLockActive, enforce_provider_allowed
            cfg = await AppConfig.aload()
            enforce_provider_allowed(session.llm_provider, cfg)
        except ImportError:
            return None
        except PrivacyLockActive:
            return {
                "type": "error",
                "content": (
                    f"Privacy Lock is engaged. This session uses **{session.llm_provider}**, "
                    "which sends data off-device and is blocked.\n\n"
                    "Start a new session with a local provider (afm, mlx, omlx, exo), "
                    "or disengage the lock in **Settings → Privacy & Security**."
                ),
                "metadata": {
                    "error_code": "privacy_lock",
                    "llm_provider": session.llm_provider,
                },
            }
        return None

    async def stream_message(
        self,
        session_id: str,
        query: str,
        context_queue: Optional[Any] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream agent responses for a user message."""
        from langchain_core.messages import HumanMessage

        session = await self._ensure_session(session_id)
        if not session:
            yield {"type": "error", "content": "Session not found"}
            return

        privacy_err = await self._check_privacy_lock(session)
        if privacy_err:
            yield privacy_err
            return

        session.message_count += 1
        session.updated_at = datetime.now(timezone.utc)
        session.status = "running"

        is_first_message = session.message_count == 1
        if is_first_message:
            session.title = query[:60] + ("..." if len(query) > 60 else "")

        await _append_message_async(session_id, {"type": "user", "content": query})
        await _transcript(session_id, "user", query, role="user")

        if session.memory_inject:
            yield {"type": "memory_search", "content": "Searching memory…"}

        run_config = {"configurable": {"thread_id": session_id}, "recursion_limit": session.recursion_limit}

        existing_state = await session.graph.aget_state({"configurable": {"thread_id": session_id}})
        existing_count = len(existing_state.values.get("messages", []))

        # First stream uses the actual user query as input.
        # After a context injection we restart with `None` so LangGraph continues
        # from the updated checkpoint (the injected HumanMessage is already in state).
        current_input: Any = {"messages": [HumanMessage(content=query)]}
        # printed_offset for the first leg: skip all historical messages + the user msg we just added.
        current_printed_offset = existing_count + 1

        while True:
            astream_iter = session.graph.astream(
                current_input,
                config=run_config,
                stream_mode="values",
            )

            context_injected = False
            injected_printed = current_printed_offset

            async for resp in self._do_stream(
                session, session_id, astream_iter,
                printed_offset=current_printed_offset,
                context_queue=context_queue,
                run_config=run_config,
            ):
                if resp.get("type") == "_context_injected":
                    # Internal signal — do not forward to the caller.
                    # Carry forward the `printed` cursor so the next leg doesn't re-emit old msgs.
                    context_injected = True
                    injected_printed = resp.get("metadata", {}).get("printed", current_printed_offset)
                else:
                    yield resp

            if not context_injected:
                break

            # Restart from the checkpoint that now contains the injected HumanMessage.
            # LangGraph continues from the last checkpoint when input is None.
            current_input = None
            current_printed_offset = injected_printed

        if is_first_message:
            await self._generate_title(session, session_id, query)

        async for resp in self._check_interrupts_or_done(session, session_id):
            yield resp

        await self._maybe_consolidate_memory(session)
        await self._maybe_index_transcript(session)
        await self._maybe_ambient_sweep(session)

    async def stream_resume(
        self,
        session_id: str,
        decisions: list[dict[str, Any]],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Resume a paused graph after a HITL decision."""
        session = await self._ensure_session(session_id)
        if not session:
            yield {"type": "error", "content": "Session not found"}
            return

        privacy_err = await self._check_privacy_lock(session)
        if privacy_err:
            yield privacy_err
            return

        run_config = {"configurable": {"thread_id": session_id}, "recursion_limit": session.recursion_limit}
        existing_state = await session.graph.aget_state({"configurable": {"thread_id": session_id}})
        existing_count = len(existing_state.values.get("messages", []))

        astream_iter = session.graph.astream(
            Command(resume={"decisions": decisions}),
            config=run_config,
            stream_mode="values",
        )

        async for resp in self._do_stream(session, session_id, astream_iter, printed_offset=existing_count):
            yield resp

        async for resp in self._check_interrupts_or_done(session, session_id):
            yield resp

        await self._maybe_consolidate_memory(session)
        await self._maybe_index_transcript(session)
        await self._maybe_ambient_sweep(session)

    async def stream_edit(
        self,
        session_id: str,
        message_index: int,
        new_content: str,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Edit a user message and replay the graph from that point."""
        from langchain_core.messages import HumanMessage

        session = await self._ensure_session(session_id)
        if not session:
            yield {"type": "error", "content": "Session not found"}
            return

        privacy_err = await self._check_privacy_lock(session)
        if privacy_err:
            yield privacy_err
            return

        run_config = {"configurable": {"thread_id": session_id}}

        target_config = None
        human_count_at_target = message_index
        async for state in session.graph.aget_state_history(run_config):
            msgs = state.values.get("messages", [])
            human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
            if len(human_msgs) == human_count_at_target:
                target_config = state.config
                break

        if not target_config:
            yield {"type": "error", "content": "Could not find checkpoint for that message"}
            return

        new_config = await session.graph.aupdate_state(
            target_config,
            {"messages": [HumanMessage(content=new_content)]},
            as_node="__start__",
        )

        all_msgs = await _load_messages_async(session_id)
        user_seen = 0
        truncate_at = 0
        for i, m in enumerate(all_msgs):
            if m.get("type") == "user":
                if user_seen == message_index:
                    truncate_at = i
                    break
                user_seen += 1
        await _save_messages_async(session_id, all_msgs[:truncate_at])
        await _append_message_async(session_id, {"type": "user", "content": new_content})
        await _transcript(session_id, "user", new_content, role="user")

        if session.memory_inject:
            yield {"type": "memory_search", "content": "Searching memory…"}

        session.updated_at = datetime.now(timezone.utc)

        forked_state = await session.graph.aget_state(new_config)
        existing_count = len(forked_state.values.get("messages", []))

        edit_run_config = dict(new_config) if isinstance(new_config, dict) else new_config
        if isinstance(edit_run_config, dict):
            edit_run_config["recursion_limit"] = session.recursion_limit

        astream_iter = session.graph.astream(
            None,
            config=edit_run_config,
            stream_mode="values",
        )

        async for resp in self._do_stream(session, session_id, astream_iter, printed_offset=existing_count):
            yield resp

        async for resp in self._check_interrupts_or_done(session, session_id):
            yield resp

        await self._maybe_consolidate_memory(session)
        await self._maybe_index_transcript(session)
        await self._maybe_ambient_sweep(session)
