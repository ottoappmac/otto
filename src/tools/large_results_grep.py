"""Regex grep over offloaded *large tool results*.

When a tool result is too big to keep in context, deepagents'
``FilesystemMiddleware`` evicts it to the virtual filesystem under
``/large_tool_results/<tool_call_id>`` and leaves only a short head/tail
preview in the conversation.  To inspect the rest, the agent normally has
two options, both unattractive:

1. ``read_file`` the offloaded blob — but the whole point of eviction is
   that the blob is large, so reading it back (even paginated) is slow and
   re-floods the context window.
2. The built-in ``grep`` tool — but it does *literal* substring matching
   only, returns no surrounding context lines, and caps results globally.

This module provides a purpose-built ``grep_large_results`` tool that:

* searches **only** under ``/large_tool_results/`` by default,
* supports full **regular expressions** (case-insensitive optional),
* returns matching lines **with a few context lines and real line
  numbers**, so the agent can follow up with a *precise*
  ``read_file(path, offset=<line-1>, limit=<n>)`` instead of reading the
  whole blob, and
* hard-caps its own output so the search result itself can never blow the
  context window.

The scan runs in-process via the backend's ``download_files`` primitive:
the full blob is read into the tool process (not the model context) and
only the (capped) matches are returned to the LLM.  This keeps the
expensive content out of the prompt entirely — that is the answer to
"read might be large".

The tool is backend-agnostic: it works with ``StateBackend`` (ephemeral
in-state files), ``FilesystemBackend``/``LocalShellBackend`` (on-disk
session files), or any other ``BackendProtocol`` implementation, because
it only relies on ``glob_info`` + ``download_files``.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool

logger = logging.getLogger(__name__)

LARGE_RESULTS_ROOT = "/large_tool_results"

# Caps so the tool's own output can never overflow the context window.
_MAX_LINE_CHARS = 500
_DEFAULT_MAX_MATCHES = 50
_HARD_MAX_MATCHES = 500
_MAX_OUTPUT_CHARS = 40_000  # ~10k tokens
# Reject pathologically long patterns before compiling.  A legitimate search
# pattern is short; a multi-kilobyte one is a sign the model fell into a
# repetition loop (the failure that motivated this guard), and compiling it
# wastes a turn and risks catastrophic regex backtracking.
_MAX_PATTERN_CHARS = 2_000

_GREP_DESCRIPTION = f"""Search OFFLOADED large tool results with a regular expression.

When a tool returns too much text, it is saved to the filesystem under
`{LARGE_RESULTS_ROOT}/<tool_call_id>` and only a short preview is shown to you.
Use THIS tool to search across those saved results without reading them whole.

Why prefer this over read_file: read_file pulls the (large) blob back into your
context. This tool scans the blob out-of-context and returns only matching lines
plus a little surrounding context, each tagged with its real line number. Use those
line numbers to do a precise read_file(path, offset=<line-1>, limit=<n>) if you need
more around a hit.

Args:
- pattern: Regular expression to search for (Python `re` syntax). Set regex=False to
  treat it as a literal string instead.
- path: Where to search. Defaults to `{LARGE_RESULTS_ROOT}` (all offloaded results).
  Pass a specific `{LARGE_RESULTS_ROOT}/<id>` to search one result.
- glob: Optional filename filter (e.g. "*.json"), matched recursively.
- output_mode: "content" (default; matching lines + context), "files_with_matches"
  (just the file paths), or "count" (matches per file).
- context: Number of context lines to show before AND after each match (default 2).
- ignore_case: Case-insensitive matching (default False).
- max_matches: Cap on matches returned (default {_DEFAULT_MAX_MATCHES}).

Examples:
- grep_large_results(pattern="error|exception", ignore_case=True)
- grep_large_results(pattern="\\"id\\":\\s*\\d+", output_mode="count")
- grep_large_results(pattern="TODO", regex=False, context=0)"""


def _resolve_backend(backend: Any, runtime: ToolRuntime) -> Any:
    """Resolve a backend instance from an instance or a runtime factory.

    Mirrors ``FilesystemMiddleware._get_backend``: ``StateBackend`` is a
    class that must be instantiated with the runtime, while on-disk
    backends are passed as ready instances.  When *backend* is ``None`` we
    fall back to the in-state ``StateBackend`` (the deepagents default).
    """
    if backend is None:
        from deepagents.backends import StateBackend

        backend = StateBackend
    return backend(runtime) if callable(backend) else backend


def _list_candidate_paths(resolved_backend: Any, path: str, glob: str | None) -> list[str]:
    """Return the virtual paths to scan under *path*.

    Handles three cases: an exact file path, a directory (recursive), and a
    missing/empty directory (returns ``[]``).
    """
    pattern = glob or "**/*"
    try:
        infos = resolved_backend.glob_info(pattern, path=path)
    except Exception as exc:  # noqa: BLE001 - backends should not throw in tools
        logger.debug("grep_large_results: glob_info failed for %s: %s", path, exc)
        infos = []

    paths = [fi.get("path", "") for fi in infos if fi.get("path")]
    if paths:
        return paths

    # `path` might itself be an exact file (e.g. a single offloaded result).
    # glob_info on a file returns nothing, so probe it directly.
    if glob is None:
        try:
            responses = resolved_backend.download_files([path])
        except Exception:  # noqa: BLE001
            return []
        if responses and responses[0].content is not None:
            return [path]
    return []


def _scan(
    resolved_backend: Any,
    paths: list[str],
    regex: "re.Pattern[str]",
    context: int,
    max_matches: int,
) -> tuple[dict[str, list[tuple[int, str, bool]]], int, bool]:
    """Scan *paths* for *regex*.

    Returns ``(results, total_matches, truncated)`` where ``results`` maps a
    file path to a list of ``(line_number, text, is_match)`` rows already
    expanded to include context lines (context rows have ``is_match=False``).
    """
    try:
        responses = resolved_backend.download_files(paths)
    except Exception as exc:  # noqa: BLE001
        logger.debug("grep_large_results: download_files failed: %s", exc)
        return {}, 0, False

    results: dict[str, list[tuple[int, str, bool]]] = {}
    total_matches = 0
    truncated = False

    for resp in responses:
        if total_matches >= max_matches:
            truncated = True
            break
        if resp.content is None:
            continue
        try:
            text = resp.content.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        lines = text.splitlines()

        # Line numbers (1-indexed) that we want to emit for this file, with
        # whether each is an actual match (vs. context).
        emit: dict[int, bool] = {}
        for idx, line in enumerate(lines):
            if regex.search(line):
                total_matches += 1
                emit[idx] = True
                for c in range(max(0, idx - context), min(len(lines), idx + context + 1)):
                    emit.setdefault(c, False)
                if total_matches >= max_matches:
                    truncated = True
                    break

        if not emit:
            continue

        rows: list[tuple[int, str, bool]] = []
        for idx in sorted(emit):
            raw = lines[idx]
            clipped = raw if len(raw) <= _MAX_LINE_CHARS else raw[:_MAX_LINE_CHARS] + " …[line truncated]"
            rows.append((idx + 1, clipped, emit[idx]))
        results[resp.path] = rows

    return results, total_matches, truncated


def _format(
    results: dict[str, list[tuple[int, str, bool]]],
    output_mode: str,
    total_matches: int,
    truncated: bool,
    root: str,
) -> str:
    if not results:
        return f"No matches found under {root}."

    if output_mode == "files_with_matches":
        body = "\n".join(sorted(results.keys()))
    elif output_mode == "count":
        body = "\n".join(
            f"{path}: {sum(1 for _, _, is_match in rows if is_match)}"
            for path, rows in sorted(results.items())
        )
    else:  # content
        blocks: list[str] = []
        for path in sorted(results.keys()):
            rows = results[path]
            n = sum(1 for _, _, is_match in rows if is_match)
            header = f"{path} ({n} match{'es' if n != 1 else ''}):"
            lines = [header]
            prev = None
            for line_num, text, is_match in rows:
                if prev is not None and line_num != prev + 1:
                    lines.append("       --")
                marker = ">" if is_match else ":"
                lines.append(f"{line_num:>6}{marker} {text}")
                prev = line_num
            blocks.append("\n".join(lines))
        body = "\n\n".join(blocks)

    footer = ""
    if truncated:
        footer = (
            f"\n\n[Showing first {total_matches} matches — refine `pattern`/`glob` or "
            f"raise `max_matches` for more. Use read_file(path, offset=<line-1>, limit=N) "
            f"to read around a specific hit.]"
        )

    out = body + footer
    if len(out) > _MAX_OUTPUT_CHARS:
        out = out[:_MAX_OUTPUT_CHARS] + "\n…[output truncated; narrow your search]"
    return out


def _run(
    resolved_backend: Any,
    pattern: str,
    path: str,
    glob: str | None,
    output_mode: str,
    context: int,
    ignore_case: bool,
    max_matches: int,
    regex: bool,
) -> str:
    if not pattern:
        return "Error: pattern must be a non-empty string."

    if len(pattern) > _MAX_PATTERN_CHARS:
        return (
            f"Error: pattern too long ({len(pattern)} chars, max {_MAX_PATTERN_CHARS}). "
            "A pattern this long usually means the request went wrong (e.g. a repeated "
            "alternation). Use a SHORT pattern that matches a keyword or ticker, or read "
            "the file directly with read_file(path, offset=<line>, limit=<n>) to page "
            "through it."
        )

    context = max(0, min(int(context), 10))
    max_matches = max(1, min(int(max_matches), _HARD_MAX_MATCHES))

    flags = re.IGNORECASE if ignore_case else 0
    try:
        compiled = re.compile(pattern if regex else re.escape(pattern), flags)
    except re.error as exc:
        return f"Error: invalid regular expression: {exc}"

    paths = _list_candidate_paths(resolved_backend, path, glob)
    if not paths:
        return (
            f"No offloaded tool results found under {path}. "
            f"Large results are saved there only after a tool returns more text than fits "
            f"in context; if you haven't triggered one, there is nothing to search."
        )

    results, total, truncated = _scan(resolved_backend, paths, compiled, context, max_matches)
    return _format(results, output_mode, total, truncated, path)


def create_grep_large_results_tool(backend: Any = None) -> BaseTool:
    """Build the ``grep_large_results`` tool bound to *backend*.

    Args:
        backend: The same backend passed to ``FilesystemMiddleware`` (an
            instance, or the ``StateBackend`` class / a factory callable).
            ``None`` defaults to the in-state ``StateBackend``.
    """

    def grep_large_results(
        pattern: str,
        runtime: ToolRuntime,
        path: str = LARGE_RESULTS_ROOT,
        glob: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "content",
        context: int = 2,
        ignore_case: bool = False,
        max_matches: int = _DEFAULT_MAX_MATCHES,
        regex: bool = True,
    ) -> str:
        resolved = _resolve_backend(backend, runtime)
        return _run(
            resolved, pattern, path, glob, output_mode,
            context, ignore_case, max_matches, regex,
        )

    async def agrep_large_results(
        pattern: str,
        runtime: ToolRuntime,
        path: str = LARGE_RESULTS_ROOT,
        glob: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "content",
        context: int = 2,
        ignore_case: bool = False,
        max_matches: int = _DEFAULT_MAX_MATCHES,
        regex: bool = True,
    ) -> str:
        import asyncio

        resolved = _resolve_backend(backend, runtime)
        return await asyncio.to_thread(
            _run, resolved, pattern, path, glob, output_mode,
            context, ignore_case, max_matches, regex,
        )

    return StructuredTool.from_function(
        name="grep_large_results",
        description=_GREP_DESCRIPTION,
        func=grep_large_results,
        coroutine=agrep_large_results,
    )
