#!/usr/bin/env python3
"""Built-in MCP server: macOS osascript.

Exposes four tools that let agents drive macOS deterministically:

* ``run_osascript``           — execute an inline script string.
* ``run_osascript_file``      — execute a script saved on disk.
* ``inspect_app_dictionary``  — return the AppleScript classes /
  commands an app exposes, so the agent can branch on whether the
  app has a real AppleScript dictionary (Mail, Notes, Music, …)
  vs. only the standard suite (Slack, Discord, Cursor, every
  Electron / Catalyst app).  Replaces LLM-side priors with a system
  probe.
* ``dump_ax_tree``            — return the Accessibility tree of an
  app's front window as text.  The agent reads the dump in its next
  call and composes System Events scripts using element paths copied
  *from the dump* instead of guessing.

This MCP is gated on macOS via ``requires_os`` in the registry — the
backend skips connection on Linux / Windows so the tools never appear
on unsupported platforms.

Trust boundaries:

* The agent can run arbitrary AppleScript / JXA — i.e. anything the
  user could run themselves with ``osascript``.  macOS already gates
  privileged surface (Accessibility, Automation per-app, Screen
  Recording, Full Disk Access) behind TCC prompts the user must
  explicitly approve, so the LLM cannot escalate beyond what the host
  binary is already entitled to.
* Per-call ``timeout_seconds`` is hard-capped at 120 s so a runaway
  script can't pin the subprocess pool.
* stdout / stderr are truncated to 64 KB so a chatty script can't blow
  past the model's context window or saturate the MCP transport.
* Only the ``AppleScript`` and ``JavaScript`` languages are accepted —
  other ``-l`` values that ``osascript`` understands historically
  (e.g. arbitrary OSA components) are rejected.

The backend never sees the script body; the LLM authors and submits it
through the standard MCP tool-call path, exactly like any other tool
parameter.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("otto.mcp.macos_osascript")


_MAX_TIMEOUT_SECS = 120
_DEFAULT_TIMEOUT_SECS = 30
_MAX_OUTPUT_BYTES = 64 * 1024  # truncate stdout / stderr beyond this
_ALLOWED_LANGUAGES = ("AppleScript", "JavaScript")


# ---------------------------------------------------------------------------
# Desktop lock — the single "talking stick" for the foreground GUI.
#
# macOS renders exactly one foreground window per login session, and
# synthesized keystrokes / clicks land on whatever is frontmost.  So any
# number of agents may run *background* automation in parallel (app
# dictionary verbs, Accessibility reads, the Mail SQLite store), but only
# ONE may drive the screen at a time.
#
# Crucially, that "one at a time" must hold across *every* agent run on the
# host, not just calls within this subprocess: each agent session spins up
# its own ``macos-osascript`` subprocess (and its own in-process
# ``macos-native`` toolkit), so an in-process ``asyncio.Lock`` would never
# serialize two concurrent runs.  The lock therefore lives in a shared,
# stdlib-only module backed by a cross-process ``fcntl.flock`` — see
# :mod:`_desktop_lock` — which the ``macos-native`` tools acquire too.
#
# Scripts that need the screen wait their turn for the lock; everything
# else skips it entirely (see :func:`_helpers.script_needs_desktop`).
# ---------------------------------------------------------------------------

# Two import paths, mirroring the ``_helpers`` import below: a bundled
# sibling module in the subprocess venv, or the package path in tests /
# the parent process.
try:
    from ._desktop_lock import (  # type: ignore[import-not-found]
        DesktopBusy as _DesktopBusy,
        LEASE_WAIT_SECS as _LEASE_WAIT_SECS,
        desktop_busy_message as _desktop_busy_message,
        desktop_lock as _desktop_access,
    )
except ImportError:
    from _desktop_lock import (  # type: ignore[no-redef]
        DesktopBusy as _DesktopBusy,
        LEASE_WAIT_SECS as _LEASE_WAIT_SECS,
        desktop_busy_message as _desktop_busy_message,
        desktop_lock as _desktop_access,
    )


def _desktop_busy_result(exc: _DesktopBusy) -> dict[str, Any]:
    """Standard error envelope for a call that never got the screen."""
    return {
        "ok": False,
        "exit_code": -1,
        "stdout": "",
        "stderr": _desktop_busy_message(exc.waited_ms),
        "stdout_truncated": False,
        "stderr_truncated": False,
        "timed_out": False,
        "duration_ms": 0,
        "desktop_busy": True,
        "desktop_waited_ms": exc.waited_ms,
    }


def _with_path_rewrites(
    result: dict[str, Any],
    rewrites: list[tuple[str, str]],
) -> dict[str, Any]:
    """Annotate a result with any virtual→real path remaps that were applied.

    Surfacing the remap (rather than doing it invisibly) lets the model see
    that ``/output/foo`` became ``$SESSION_FILES/output/foo`` — useful when a
    later step references the same path or when debugging an attachment.
    """
    if rewrites:
        result["path_rewrites"] = [
            {"from": original, "to": replacement}
            for original, replacement in rewrites
        ]
    return result


mcp = FastMCP("macOS osascript")


def _truncate(blob: bytes) -> tuple[str, bool]:
    """Decode subprocess output, truncating to ``_MAX_OUTPUT_BYTES``.

    Returns ``(text, truncated)`` so callers can surface a "[…truncated]"
    suffix when the LLM might otherwise miss that the script kept
    printing past the cap.
    """
    truncated = len(blob) > _MAX_OUTPUT_BYTES
    if truncated:
        blob = blob[:_MAX_OUTPUT_BYTES]
    try:
        return blob.decode("utf-8", errors="replace"), truncated
    except Exception:
        return blob.decode("latin-1", errors="replace"), truncated


def _validate_language(language: str) -> str:
    """Normalize and validate the OSA language flag."""
    if language not in _ALLOWED_LANGUAGES:
        raise ValueError(
            f"language must be one of {list(_ALLOWED_LANGUAGES)}; got {language!r}"
        )
    return language


def _validate_timeout(timeout: int) -> int:
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    if timeout > _MAX_TIMEOUT_SECS:
        raise ValueError(
            f"timeout_seconds capped at {_MAX_TIMEOUT_SECS} for safety; "
            f"got {timeout}"
        )
    return timeout


async def _run_osascript_args(args: list[str], timeout: int) -> dict[str, Any]:
    """Spawn ``osascript`` and collect its result.

    Uses ``create_subprocess_exec`` (no shell) so the LLM's script body
    is delivered as a single argv entry — no shell metachar escaping
    needed and no risk of chaining unrelated commands via ``;`` / ``&&``.
    """
    started = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "osascript binary not found — is this running on macOS?",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 0,
        }

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        timed_out = True
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except Exception:
            stdout_bytes, stderr_bytes = b"", b""

    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_text, stdout_truncated = _truncate(stdout_bytes or b"")
    stderr_text, stderr_truncated = _truncate(stderr_bytes or b"")

    exit_code = proc.returncode if proc.returncode is not None else -1

    return {
        "ok": (not timed_out) and exit_code == 0,
        "exit_code": exit_code,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
    }


@mcp.tool()
async def run_osascript(
    script: str,
    language: str = "AppleScript",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Execute an AppleScript or JXA snippet via ``osascript -e``.

    Use this whenever you need to drive macOS at the OSA level —
    activating apps, dispatching keystrokes via System Events,
    controlling iTunes / Music / Safari / Mail, querying System
    Preferences, or running anything from the AppleScript dictionary
    of an installed app.

    Prefer the ``macos-native`` Accessibility tools when the goal is
    UI automation by element index — those are deterministic and don't
    require knowing the target app's AppleScript dictionary.  Reach
    for ``run_osascript`` when:

    * The app exposes useful AppleScript verbs (``tell application "X" to …``).
    * You need to read or set a system-level property (volume, display,
      brightness, idle time, …).
    * You're integrating with Shortcuts, Automator, or another OSA
      consumer.

    **File paths inside the script**: this script runs on the *host*, so a
    virtual session path embedded in a ``POSIX file "…"`` literal (e.g. a
    Mail ``make new attachment`` referencing ``/output/report.pdf``) is
    auto-remapped to the real ``$SESSION_FILES`` sandbox path before
    execution.  You can therefore write ``POSIX file "/output/<name>"`` and
    it will resolve to the file ``write_file`` saved.  Only ``POSIX file``
    literals are remapped — free-text fields like an email ``content`` /
    ``subject`` are left untouched.  Any remaps applied are reported back in
    the ``path_rewrites`` key of the result.

    Args:
        script: AppleScript or JXA source.  Multi-line scripts are
                fine — pass the body verbatim.
        language: ``"AppleScript"`` (default) or ``"JavaScript"`` (JXA).
                  Case-sensitive; anything else is rejected.
        timeout_seconds: Wall-clock cap.  Capped at 120 s.  Defaults to 30 s.

    Concurrency: scripts that steal focus or synthesize input
    (``activate``, ``keystroke``, ``key code``, ``click``, …) acquire a
    process-wide *desktop lease* so only one agent drives the screen at a
    time; other agents queue.  Background scripts (app dictionary verbs,
    read-only Accessibility queries) skip the lease and run in parallel.
    Prefer background approaches when possible.

    Returns:
        dict with ``ok``, ``exit_code``, ``stdout``, ``stderr``,
        ``stdout_truncated``, ``stderr_truncated``, ``timed_out``,
        ``duration_ms``.  ``stdout`` / ``stderr`` are truncated to 64 KB.
        Screen-driving calls also include ``desktop_waited_ms`` (time
        spent queued for the screen).  If the screen never freed up the
        call returns ``ok=False`` with ``desktop_busy=True``.
    """
    if not isinstance(script, str) or not script.strip():
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "script must be a non-empty string",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 0,
        }

    try:
        lang = _validate_language(language)
        timeout = _validate_timeout(timeout_seconds)
    except ValueError as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 0,
        }

    # Inline scripts run on the host, so a virtual session path embedded in
    # a ``POSIX file "/output/…"`` literal would resolve against the host
    # filesystem root instead of the sandbox where the agent saved the file
    # (the classic "email sent without the attachment" bug).  Remap those
    # literals to the real ``$SESSION_FILES`` path before executing.
    script, path_rewrites = _rewrite_virtual_posix_paths(script)

    args = ["osascript", "-l", lang, "-e", script]

    # Background scripts (dictionary verbs, read-only AX queries) run
    # concurrently; only focus-stealing / input-synthesizing scripts take
    # the desktop lease so they can't collide with another agent.
    if not _script_needs_desktop(script):
        result = await _run_osascript_args(args, timeout=timeout)
        return _with_path_rewrites(result, path_rewrites)

    try:
        async with _desktop_access() as waited_ms:
            result = await _run_osascript_args(args, timeout=timeout)
    except _DesktopBusy as exc:
        return _desktop_busy_result(exc)
    result["desktop_waited_ms"] = waited_ms
    return _with_path_rewrites(result, path_rewrites)


def _resolve_script_path(path: str) -> Path:
    """Resolve a script path to a real host filesystem path.

    Agents write files through a virtual filesystem where ``/`` maps to the
    session sandbox (``$SESSION_FILES``).  When a path starts with
    ``/output/`` or ``/`` without a real host directory, remap it to
    ``$SESSION_FILES<path>`` so osascript — which runs on the host — can
    actually find the file.

    Resolution order:
    1. ``~``-expansion and ``resolve()`` — if the result is an existing file,
       use it as-is (real absolute paths like ``/Users/…`` or
       ``$SESSION_FILES/…`` already expanded by the caller).
    2. If the resolved path does not exist AND ``SESSION_FILES`` is set in the
       environment, try ``$SESSION_FILES<path>`` (virtual-root translation).
    3. Return the best candidate so the caller can surface a clear error.
    """
    expanded = Path(os.path.expanduser(path)).resolve()
    if expanded.is_file():
        return expanded
    session_files = os.environ.get("SESSION_FILES", "")
    if session_files:
        candidate = Path(session_files) / path.lstrip("/")
        if candidate.is_file():
            return candidate
        # Also try path as-is relative to SESSION_FILES (handles /output/foo)
        candidate2 = Path(session_files + path)
        if candidate2.is_file():
            return candidate2
    return expanded  # callers check .is_file() for a clear error message


@mcp.tool()
async def run_osascript_file(
    path: str,
    language: str = "AppleScript",
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Execute an ``.applescript`` / ``.scpt`` / ``.js`` file via osascript.

    Use this when the script is large or reusable and lives on disk.  The
    interpreter is selected by the ``language`` flag — same options as
    ``run_osascript``.

    **Path conventions** — the agent's file tools use a virtual filesystem
    where ``/`` is the session sandbox.  ``run_osascript_file`` runs on the
    *host*, so it accepts both real absolute paths and virtual session paths:

    * Real host path: ``/Users/alice/scripts/foo.applescript``
    * ``$SESSION_FILES``-expanded path: pass the value of ``$SESSION_FILES``
      directly — e.g. ``/Users/…/Otto/sessions/<id>/files/foo.applescript``.
    * Virtual session path: ``/output/foo.applescript`` — automatically
      remapped to ``$SESSION_FILES/output/foo.applescript``.

    Prefer writing scripts with ``execute`` into ``$SESSION_FILES`` and
    passing that real path here, rather than using ``/output/…`` virtual
    paths which require the remapping step.

    Args:
        path: Absolute, ``~``-relative, or virtual ``/output/…`` path to the
              script file.
        language: ``"AppleScript"`` (default) or ``"JavaScript"`` (JXA).
                  Case-sensitive; anything else is rejected.
        timeout_seconds: Wall-clock cap.  Capped at 120 s.  Defaults to 30 s.

    Returns:
        Same shape as ``run_osascript``.
    """
    expanded = _resolve_script_path(path)
    if not expanded.is_file():
        session_files = os.environ.get("SESSION_FILES", "")
        hint = (
            f"  SESSION_FILES={session_files!r}" if session_files else
            "  SESSION_FILES not set — pass an absolute host path instead."
        )
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": (
                f"script file not found: {expanded}\n"
                f"Tip: use the real $SESSION_FILES path, not the virtual /output/ path.\n"
                + hint
            ),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 0,
        }

    try:
        lang = _validate_language(language)
        timeout = _validate_timeout(timeout_seconds)
    except ValueError as exc:
        return {
            "ok": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "stdout_truncated": False,
            "stderr_truncated": False,
            "timed_out": False,
            "duration_ms": 0,
        }

    # Peek at the script body to (a) decide whether it needs the screen and
    # (b) remap any virtual ``POSIX file "/output/…"`` literals to the real
    # sandbox path — the file runs on the host, so an unremapped virtual path
    # would resolve against the host root.  A compiled ``.scpt`` is binary, so
    # the text read fails: when in doubt (unreadable file) we run it by path
    # and take the lease, since over-serializing is safe but a missed
    # collision is not.
    needs_desktop = True
    path_rewrites: list[tuple[str, str]] = []
    args = ["osascript", "-l", lang, str(expanded)]
    try:
        body = expanded.read_text(encoding="utf-8", errors="replace")
    except OSError:
        body = None
    if body is not None:
        rewritten, path_rewrites = _rewrite_virtual_posix_paths(body)
        needs_desktop = _script_needs_desktop(rewritten)
        # Only switch to inline ``-e`` execution when we actually changed a
        # path; otherwise run the file by path (cheaper, preserves line
        # numbers in any error output).
        if path_rewrites:
            args = ["osascript", "-l", lang, "-e", rewritten]

    if not needs_desktop:
        result = await _run_osascript_args(args, timeout=timeout)
        return _with_path_rewrites(result, path_rewrites)

    try:
        async with _desktop_access() as waited_ms:
            result = await _run_osascript_args(args, timeout=timeout)
    except _DesktopBusy as exc:
        return _desktop_busy_result(exc)
    result["desktop_waited_ms"] = waited_ms
    return _with_path_rewrites(result, path_rewrites)


# ---------------------------------------------------------------------------
# Introspection tools — replace LLM-side priors with system probes.
#
# Two failure modes motivate these:
#
# 1. The LLM "knows" Slack has a dictionary because it has a ``version``.
#    It doesn't.  Slack / Discord / Cursor / VS Code / Linear / Figma /
#    every Electron app responds to the standard suite (``version``,
#    ``activate``, ``quit``) and nothing else.  ``inspect_app_dictionary``
#    surfaces the truth in one call so the agent can branch on
#    ``has_app_specific_suite``.
# 2. The LLM guesses System Events element paths
#    (``click "Direct Messages"`` — invalid; ``click button "Direct
#    Messages" of group 2 of window 1`` — valid) because it can't see
#    the AX tree.  ``dump_ax_tree`` returns the tree as text so the
#    agent reads element paths instead of inventing them.
# ---------------------------------------------------------------------------


# Pure-Python helpers live in a sibling module so unit tests can import
# them without going through the @mcp.tool() decorator (which has
# version-sensitive parameter introspection in newer FastMCP).  At
# runtime the MCP server lives in its own uv-provisioned venv where
# the decorator works.
#
# Two import paths are supported because this file runs in two contexts:
#
# * Production: spawned as ``python <path-to-server.py>`` (see
#   :mod:`backend.builtin_mcps.registry`), so the script's directory
#   is on ``sys.path`` and ``_helpers`` resolves as a top-level module.
# * Tests / parent process: imported as
#   ``backend.builtin_mcps.macos_osascript.server`` via the implicit
#   namespace package, where the relative import works.
try:
    from ._helpers import (  # type: ignore[import-not-found]
        classify_dictionary as _classify_dictionary,
        escape_applescript_string as _escape_applescript_string,
        resolve_app_path as _resolve_app_path,
        rewrite_virtual_posix_paths as _rewrite_virtual_posix_paths,
        script_needs_desktop as _script_needs_desktop,
    )
except ImportError:
    from _helpers import (  # type: ignore[no-redef]
        classify_dictionary as _classify_dictionary,
        escape_applescript_string as _escape_applescript_string,
        resolve_app_path as _resolve_app_path,
        rewrite_virtual_posix_paths as _rewrite_virtual_posix_paths,
        script_needs_desktop as _script_needs_desktop,
    )


@mcp.tool()
async def inspect_app_dictionary(app_name: str) -> dict[str, Any]:
    """Return the AppleScript dictionary classes and commands for an app.

    Use this BEFORE composing any ``tell application "<X>" to …``
    script.  The result tells you whether the app exposes a real
    AppleScript dictionary (``has_app_specific_suite=True`` — Mail,
    Notes, Music, Calendar, Reminders, Safari, Finder, …) or only the
    standard suite (``has_app_specific_suite=False`` — Slack, Discord,
    Cursor, VS Code, Linear, Figma, Zoom, Teams, Obsidian, Notion, and
    every other Electron / Catalyst app).

    Branching rule:

    * ``has_app_specific_suite=True`` and ``classes`` contains the noun
      you need (``message``, ``note``, ``track``, ``tab``, …) →
      compose a dictionary-driven script.
    * Otherwise → call ``dump_ax_tree`` instead and drive via System
      Events using paths copied from the dump.

    CRITICAL: read ``properties[<noun>]`` for the real field names before
    scripting (e.g. ``properties["message"]``).  Do NOT guess property
    names — dictionaries vary by app (Apple Mail uses ``date received``;
    Microsoft Outlook uses ``time received`` and even defines ``sender``
    as a *class*, so ``sender of msg`` raises ``-2741 "found class
    name"``).  Use only names that appear in the ``properties`` map.

    **Microsoft Outlook Exchange warning**: Outlook for Mac v16.x ("New
    Outlook") stores Exchange emails in a proprietary ``HxStore.hxd``
    binary, **not** in the AppleScript-accessible ``inbox`` folder.
    ``messages of inbox`` (and ``folder "Inbox"``) will always return an
    empty list even when the UI shows unread mail.  If the goal is to
    check Outlook unread count, call ``query_outlook_store`` instead —
    it reads the ``ExternalCounters.ctr`` file that Outlook keeps current
    with the true Exchange badge count.

    Args:
        app_name: User-facing app name (e.g. ``"Slack"``, ``"Mail"``,
            ``"Cursor"``).  Must match the ``.app`` bundle name on disk.

    Returns:
        dict with:
        * ``ok``                       — bool
        * ``has_app_specific_suite``   — False for Electron / Catalyst
        * ``classes``                  — app-specific class names
                                          (standard-suite names stripped)
        * ``commands``                 — exposed verbs
        * ``properties``               — map of class name → its property
                                          names (from ``<class>`` and
                                          ``<class-extension>`` blocks);
                                          use these instead of guessing
        * ``elements``                 — map of class name → the ``type``
                                          names of collections it contains
                                          (e.g. ``mailbox`` → ``message``
                                          ⇒ ``messages of mailbox``); use
                                          to navigate, not ``items of``
        * ``app_path``                 — resolved bundle path or None
        * ``raw_sdef``                 — sdef XML, truncated 64 KB
        * ``stderr``                   — error message on failure
    """
    if not isinstance(app_name, str) or not app_name.strip():
        return {
            "ok": False,
            "has_app_specific_suite": False,
            "classes": [],
            "commands": [],
            "properties": {},
            "elements": {},
            "app_path": None,
            "raw_sdef": "",
            "stderr": "app_name must be a non-empty string",
        }

    app_path = _resolve_app_path(app_name)
    if app_path is None:
        return {
            "ok": False,
            "has_app_specific_suite": False,
            "classes": [],
            "commands": [],
            "properties": {},
            "elements": {},
            "app_path": None,
            "raw_sdef": "",
            "stderr": (
                f"could not find {app_name}.app under /Applications, "
                "/System/Applications, or ~/Applications. Pass the exact "
                "name as it appears in Finder."
            ),
        }

    res = await _run_osascript_args(
        ["sdef", app_path],
        timeout=30,
    )
    if not res.get("ok"):
        return {
            "ok": False,
            "has_app_specific_suite": False,
            "classes": [],
            "commands": [],
            "properties": {},
            "elements": {},
            "app_path": app_path,
            "raw_sdef": "",
            "stderr": (
                f"`sdef` failed for {app_path}: "
                + (res.get("stderr") or res.get("stdout") or "unknown error")
            ),
        }

    sdef_xml = res.get("stdout") or ""
    has_specific, classes, commands, properties, elements = _classify_dictionary(
        sdef_xml
    )

    return {
        "ok": True,
        "has_app_specific_suite": has_specific,
        "classes": classes,
        "commands": commands,
        "properties": properties,
        "elements": elements,
        "app_path": app_path,
        "raw_sdef": sdef_xml,
        "stderr": "",
    }


@mcp.tool()
async def dump_ax_tree(
    app_name: str,
    *,
    window_index: int = 1,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECS,
) -> dict[str, Any]:
    """Dump an app's front-window Accessibility tree as text.

    Use this when ``inspect_app_dictionary`` reported
    ``has_app_specific_suite=False`` (Electron / Catalyst app), or when
    a dictionary-driven script returned ``-2741`` / ``-1719`` because
    the class or object you guessed doesn't exist.

    The dump is the deterministic source of truth for element paths.
    Read it in your next call and compose your System Events script
    using paths COPIED from the dump (``button "Send" of window 1``,
    ``static text 1 of group 2 of window 1``, …).  Never guess paths.

    Wraps:

    .. code-block:: applescript

        tell application "<app>" to activate
        delay 0.3
        tell application "System Events"
            tell process "<app>"
                return entire contents of window <window_index>
            end tell
        end tell

    Args:
        app_name: User-facing app name (must match the running process).
        window_index: 1-based index of the target window.  Most apps
            have one main window; multi-window apps (Mail, Cursor) may
            need ``2`` or higher.
        timeout_seconds: Wall-clock cap.  Capped at 120 s.  Defaults to 30 s.

    Returns:
        dict with:
        * ``ok``                — bool
        * ``tree``              — text dump of ``entire contents``
                                   (truncated to 64 KB)
        * ``tree_truncated``    — bool; True when the dump was clipped
        * ``stderr``            — error message on failure
        * ``timed_out``         — bool
        * ``duration_ms``       — int
    """
    if not isinstance(app_name, str) or not app_name.strip():
        return {
            "ok": False,
            "tree": "",
            "tree_truncated": False,
            "stderr": "app_name must be a non-empty string",
            "timed_out": False,
            "duration_ms": 0,
        }

    try:
        timeout = _validate_timeout(timeout_seconds)
    except ValueError as exc:
        return {
            "ok": False,
            "tree": "",
            "tree_truncated": False,
            "stderr": str(exc),
            "timed_out": False,
            "duration_ms": 0,
        }

    if not isinstance(window_index, int) or window_index < 1:
        return {
            "ok": False,
            "tree": "",
            "tree_truncated": False,
            "stderr": "window_index must be a positive integer",
            "timed_out": False,
            "duration_ms": 0,
        }

    safe_name = _escape_applescript_string(app_name)
    script = (
        f'tell application "{safe_name}" to activate\n'
        f"delay 0.3\n"
        f'tell application "System Events"\n'
        f'    tell process "{safe_name}"\n'
        f"        return entire contents of window {window_index} as text\n"
        f"    end tell\n"
        f"end tell"
    )

    # ``dump_ax_tree`` always ``activate``s the target app to guarantee
    # window 1 is the one being read, so it always takes the lease.
    try:
        async with _desktop_access() as waited_ms:
            res = await _run_osascript_args(
                ["osascript", "-l", "AppleScript", "-e", script],
                timeout=timeout,
            )
    except _DesktopBusy as exc:
        busy = _desktop_busy_result(exc)
        return {
            "ok": False,
            "tree": "",
            "tree_truncated": False,
            "stderr": busy["stderr"],
            "timed_out": False,
            "duration_ms": 0,
            "desktop_busy": True,
            "desktop_waited_ms": exc.waited_ms,
        }

    return {
        "ok": bool(res.get("ok")),
        "tree": res.get("stdout") or "",
        "tree_truncated": bool(res.get("stdout_truncated")),
        "stderr": res.get("stderr") or "",
        "timed_out": bool(res.get("timed_out")),
        "duration_ms": int(res.get("duration_ms") or 0),
        "desktop_waited_ms": waited_ms,
    }


@mcp.tool()
async def query_mail_store(
    *,
    account_email: str = "",
    keywords: list[str] | None = None,
    days_back: int = 42,
    mailbox_name: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    """Query Apple Mail's local SQLite metadata store without Apple Events.

    Apple Mail caches all message metadata (subject, sender, date, mailbox)
    in a SQLite database at ``~/Library/Mail/V*/MailData/Envelope Index``.
    Reading this directly is **orders of magnitude faster** than iterating
    messages via AppleScript, because it avoids per-message Apple Event
    round-trips and IMAP network fetches.

    Use this tool FIRST when the goal is to **search or filter** Mail
    messages (e.g. "find emails about X in the last N weeks").  Fall back
    to ``run_osascript`` with Mail's dictionary only for tasks that require
    *fetching email bodies* or *taking actions* (reply, move, delete).

    **Important**: requires ``Full Disk Access`` for the running process.
    If the call returns ``ok=false`` with a permissions error, surface the
    TCC message to the user — they must grant Full Disk Access in System
    Settings → Privacy & Security.

    Args:
        account_email: Filter to a specific account address (e.g.
            ``"alice@gmail.com"``).  Empty string matches all accounts.
        keywords: List of substrings to match (case-insensitive OR across
            subject and sender).  Empty list / ``None`` returns all recent
            messages (up to ``limit``).
        days_back: How many calendar days to look back.  Default 42 (6 weeks).
        mailbox_name: Filter by mailbox name substring (e.g. ``"INBOX"``,
            ``"Sent"``).  Empty string matches all mailboxes.
        limit: Maximum rows to return.  Default 100.  Cap at 1000.

    Returns:
        dict with:
        * ``ok``        — bool
        * ``rows``      — list of dicts with keys ``subject``, ``sender``,
                           ``date_received`` (ISO-8601), ``mailbox``,
                           ``account``
        * ``count``     — number of rows returned
        * ``db_path``   — path to the database file used
        * ``stderr``    — error message on failure
    """
    import sqlite3
    import glob as _glob
    from datetime import datetime, timezone, timedelta

    limit = min(max(1, limit), 1000)

    # ── Find the Envelope Index ───────────────────────────────────────────
    mail_base = Path(os.path.expanduser("~/Library/Mail"))
    pattern = str(mail_base / "V*" / "MailData" / "Envelope Index")
    candidates = sorted(_glob.glob(pattern), reverse=True)  # newest version first

    if not candidates:
        return {
            "ok": False,
            "rows": [],
            "count": 0,
            "db_path": "",
            "stderr": (
                "Apple Mail database not found at ~/Library/Mail/V*/MailData/Envelope Index. "
                "Either Mail has never been opened or Full Disk Access is not granted — "
                "grant it in System Settings → Privacy & Security → Full Disk Access."
            ),
        }

    db_path = candidates[0]

    # ── Build the query ───────────────────────────────────────────────────
    cutoff_ts = (
        datetime.now(timezone.utc) - timedelta(days=days_back)
    ).timestamp()

    # The Envelope Index stores date_received as a Mac absolute time
    # (seconds since 2001-01-01 00:00:00 UTC), not Unix epoch.
    # Offset: 978307200 seconds between 1970-01-01 and 2001-01-01.
    _COCOA_EPOCH_OFFSET = 978307200
    cutoff_cocoa = cutoff_ts - _COCOA_EPOCH_OFFSET

    try:
        conn = await asyncio.to_thread(
            sqlite3.connect, db_path, timeout=10.0,
        )
        conn.row_factory = sqlite3.Row

        # Build WHERE clauses
        conditions: list[str] = ["m.date_received >= :cutoff"]
        params: dict[str, Any] = {"cutoff": cutoff_cocoa, "limit": limit}

        if mailbox_name:
            conditions.append("lower(mb.url) LIKE :mbname OR lower(mb.name) LIKE :mbname")
            params["mbname"] = f"%{mailbox_name.lower()}%"

        if account_email:
            conditions.append(
                "lower(mb.url) LIKE :acct OR lower(a.account_name) LIKE :acct"
            )
            params["acct"] = f"%{account_email.lower()}%"

        where = " AND ".join(conditions)

        sql = f"""
            SELECT
                m.subject,
                m.sender,
                m.date_received,
                mb.name AS mailbox,
                mb.url  AS mailbox_url
            FROM messages m
            LEFT JOIN mailboxes mb ON m.mailbox = mb.rowid
            WHERE {where}
            ORDER BY m.date_received DESC
            LIMIT :limit
        """

        def _run_query() -> list[sqlite3.Row]:
            with conn:
                return conn.execute(sql, params).fetchall()

        rows = await asyncio.to_thread(_run_query)
        conn.close()

    except sqlite3.OperationalError as exc:
        err = str(exc)
        if "unable to open" in err.lower() or "permission" in err.lower():
            err = (
                f"Cannot open {db_path}: {exc}. "
                "Grant Full Disk Access in System Settings → Privacy & Security."
            )
        return {"ok": False, "rows": [], "count": 0, "db_path": db_path, "stderr": err}
    except Exception as exc:
        return {"ok": False, "rows": [], "count": 0, "db_path": db_path, "stderr": str(exc)}

    # ── Filter by keywords (post-query, subject + sender) ─────────────────
    kws = [k.lower() for k in (keywords or []) if k]
    result_rows = []
    for row in rows:
        subject = (row["subject"] or "").lower()
        sender = (row["sender"] or "").lower()
        combined = subject + " " + sender
        if kws and not any(k in combined for k in kws):
            continue
        # Convert Cocoa timestamp → ISO-8601
        cocoa_ts = row["date_received"] or 0
        unix_ts = cocoa_ts + _COCOA_EPOCH_OFFSET
        try:
            dt_str = datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            dt_str = ""
        result_rows.append({
            "subject": row["subject"] or "",
            "sender": row["sender"] or "",
            "date_received": dt_str,
            "mailbox": row["mailbox"] or "",
            "mailbox_url": row["mailbox_url"] or "",
        })

    return {
        "ok": True,
        "rows": result_rows,
        "count": len(result_rows),
        "db_path": db_path,
        "stderr": "",
    }


@mcp.tool()
async def query_outlook_store(
    *,
    profile_name: str = "Main Profile",
) -> dict[str, Any]:
    """Return the unread email count from the Microsoft Outlook Exchange inbox.

    Microsoft Outlook for Mac (v16.x "New Outlook") stores Exchange emails in a
    proprietary binary ``HxStore.hxd`` file, **not** in the AppleScript-accessible
    ``inbox`` folder.  As a result, ``tell application "Microsoft Outlook" to get
    messages of inbox`` always returns an empty list for Exchange-backed accounts,
    making the standard AppleScript path unreliable.

    This tool reads Outlook's ``ExternalCounters.ctr`` file instead — a small
    binary file that Outlook keeps up-to-date with the current Exchange unread
    badge count (the same number shown in the Dock and the sidebar).

    **When to use**:
    - To check how many unread emails the user has in their Outlook Exchange inbox.
    - After a trigger fires with an unread count, to confirm the count is still
      accurate before composing a notification.

    **Limitations**:
    - Returns only the *count*, not message details (sender, subject, time).
    - Only works for the "New Outlook" experience (v16.x+).  Classic Outlook
      users should use ``run_osascript`` with the Outlook AppleScript dictionary.
    - The counter file path depends on the Office container group ID
      (``UBF8T346G9.Office``), which is stable across Office versions.

    Args:
        profile_name: Outlook profile folder name inside ``Outlook 15 Profiles/``.
                      Defaults to ``"Main Profile"``.

    Returns:
        dict with:
        * ``ok``            — bool
        * ``unread_count``  — int; -1 if the file could not be read
        * ``counter_path``  — str; absolute path to the ExternalCounters file used
        * ``stderr``        — error message on failure
    """
    import struct
    import glob as _glob

    # The Office container group ID is stable across Office versions on macOS.
    base = Path(os.path.expanduser(
        f"~/Library/Group Containers/UBF8T346G9.Office/Outlook/"
        f"Outlook 15 Profiles/{profile_name}/ExternalCounters.ctr"
    ))

    if not base.exists():
        # Try to discover the profile automatically if the default doesn't exist.
        pattern = str(Path(os.path.expanduser(
            "~/Library/Group Containers/UBF8T346G9.Office/Outlook/"
            "Outlook 15 Profiles/*/ExternalCounters.ctr"
        )))
        candidates = sorted(_glob.glob(pattern))
        if not candidates:
            return {
                "ok": False,
                "unread_count": -1,
                "counter_path": str(base),
                "stderr": (
                    f"ExternalCounters.ctr not found at {base}. "
                    "Ensure Microsoft Outlook (New Outlook) has been opened at least once "
                    "and an Exchange account is configured."
                ),
            }
        base = Path(candidates[0])

    try:
        data = await asyncio.to_thread(base.read_bytes)
    except OSError as exc:
        return {
            "ok": False,
            "unread_count": -1,
            "counter_path": str(base),
            "stderr": f"Could not read {base}: {exc}",
        }

    # File format: 4-byte magic "XENO", 4-byte checksum/version, then uint32_le
    # unread count at offset 8.  The file is typically 48 bytes.
    if len(data) < 12 or data[:4] != b"XENO":
        return {
            "ok": False,
            "unread_count": -1,
            "counter_path": str(base),
            "stderr": (
                f"Unexpected file format (expected XENO header, got {data[:4]!r}). "
                "Outlook may have changed its counter format."
            ),
        }

    unread_count = struct.unpack_from("<I", data, 8)[0]
    return {
        "ok": True,
        "unread_count": unread_count,
        "counter_path": str(base),
        "stderr": "",
    }


if __name__ == "__main__":
    mcp.run()
