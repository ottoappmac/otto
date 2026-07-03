"""Pure-Python helpers for the macos-osascript MCP.

Lives in a separate module so unit tests can import the logic without
triggering the ``@mcp.tool()`` decorator, which fails to validate
parameter signatures under some MCP / FastMCP versions in the parent
process's test env.  At runtime the MCP server runs in its own
uv-provisioned venv (see :mod:`backend.builtin_mcps.registry`) where
the decorator works fine.

Two responsibilities:

* :func:`classify_dictionary` — parse an ``sdef`` XML dump and decide
  whether the app exposes a real AppleScript dictionary or only the
  standard suite (Electron / Catalyst case).  Used to replace LLM-side
  priors with a deterministic system probe.
* :func:`resolve_app_path` — find the ``.app`` bundle for a user-facing
  app name across the conventional install locations.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional


# Heuristics for the sdef output: when none of these tokens appear, the
# app exposes only the standard suite (no app-specific dictionary).
_APP_SPECIFIC_SUITE_TOKENS: tuple[str, ...] = (
    "<class ", "<command ", "<element ", "<property ",
)


# Standard-suite class names appear in every scriptable app's sdef
# output; their presence alone does not mean the app has its own
# dictionary.  Strip them from the result so the agent sees the real
# delta.
_STANDARD_CLASSES: frozenset[str] = frozenset({
    "application", "color", "document", "item", "window",
    "rich text", "character", "paragraph", "word", "attribute run",
    "attachment", "text", "print settings",
})


_CLASS_RE = re.compile(r'<class\s+name="([^"]+)"')
_COMMAND_RE = re.compile(r'<command\s+name="([^"]+)"')
_PROPERTY_RE = re.compile(r'<property\s+[^>]*?name="([^"]+)"')
_ELEMENT_RE = re.compile(r'<element\s+[^>]*?type="([^"]+)"')

# A ``<class>`` / ``<class-extension>`` block: capture the owning class
# name (``name="…"`` for a class, ``extends="…"`` for an extension) and
# the block body up to the matching close tag.  Apps like Microsoft
# Outlook declare most of a class's properties on a separate
# ``<class-extension extends="message">`` block, so both shapes must be
# walked or the property list comes back half-empty.
_CLASS_BLOCK_RE = re.compile(
    r'<class\s+[^>]*?name="([^"]+)"[^>]*?>(.*?)</class>',
    re.DOTALL,
)
_CLASS_EXTENSION_BLOCK_RE = re.compile(
    r'<class-extension\s+[^>]*?extends="([^"]+)"[^>]*?>(.*?)</class-extension>',
    re.DOTALL,
)


def classify_dictionary(
    sdef_xml: str,
) -> tuple[bool, list[str], list[str], dict[str, list[str]], dict[str, list[str]]]:
    """Pull class / command / property / element names out of an ``sdef``
    XML dump.

    Regex-based and tolerant — sdef output is well-structured but we
    don't want to take a hard dependency on ``xml.etree`` choking on
    entity references in vendor-shipped dictionaries.

    Returns:
        ``(has_app_specific_suite, classes, commands, properties,
        elements)`` where ``classes`` excludes the standard-suite names,
        ``properties`` maps each class name (including standard ones,
        since e.g. ``message`` may be extended) to its sorted property
        names, and ``elements`` maps each class name to the sorted
        ``type`` names of the collections it contains (e.g. ``mailbox``
        contains ``message`` → the agent knows ``messages of mailbox``
        is the right accessor instead of guessing ``items of``).
        Properties and elements are both collected from ``<class>`` and
        ``<class-extension extends="…">`` blocks so the agent sees the
        real, app-specific shape instead of guessing.
    """
    has_specific_token = any(tok in sdef_xml for tok in _APP_SPECIFIC_SUITE_TOKENS)
    classes = sorted(set(_CLASS_RE.findall(sdef_xml)))
    commands = sorted(set(_COMMAND_RE.findall(sdef_xml)))

    props: dict[str, set[str]] = {}
    elems: dict[str, set[str]] = {}
    for block_re in (_CLASS_BLOCK_RE, _CLASS_EXTENSION_BLOCK_RE):
        for owner, body in block_re.findall(sdef_xml):
            prop_names = _PROPERTY_RE.findall(body)
            if prop_names:
                props.setdefault(owner, set()).update(prop_names)
            elem_names = _ELEMENT_RE.findall(body)
            if elem_names:
                elems.setdefault(owner, set()).update(elem_names)
    properties = {owner: sorted(names) for owner, names in props.items()}
    elements = {owner: sorted(names) for owner, names in elems.items()}

    app_classes = [c for c in classes if c not in _STANDARD_CLASSES]
    has_app_specific = has_specific_token and bool(app_classes or commands)

    return has_app_specific, app_classes, commands, properties, elements


def resolve_app_path(app_name: str) -> Optional[str]:
    """Find the ``.app`` bundle for ``app_name`` across the conventional
    install locations: ``/Applications``, ``/System/Applications``,
    ``/System/Applications/Utilities``, and ``~/Applications``.

    Returns the absolute path string when one of the candidates exists,
    or ``None`` when the app isn't installed in any of the standard
    locations.  Callers should surface ``None`` to the LLM with a clear
    error message rather than failing silently.
    """
    candidates = [
        Path("/Applications") / f"{app_name}.app",
        Path("/System/Applications") / f"{app_name}.app",
        Path("/System/Applications/Utilities") / f"{app_name}.app",
        Path.home() / "Applications" / f"{app_name}.app",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def escape_applescript_string(value: str) -> str:
    """Escape a Python string for embedding in an AppleScript string literal.

    AppleScript escapes embedded double-quotes with a backslash and
    treats backslashes themselves as literal — the standard escape
    rules apply.  Used by ``dump_ax_tree`` to compose its System Events
    script body without code-injection from the agent's ``app_name``.
    """
    return value.replace("\\", "\\\\").replace('"', '\\"')


# Verbs that take over the single foreground GUI session: they either
# raise an app to the front (stealing focus from whatever the user — or
# another agent — was driving) or synthesize keyboard / mouse input,
# which is delivered to *whatever* is frontmost at that instant.  Any
# script containing one of these must run under the desktop lease so two
# agents can never drive the screen at the same moment.
#
# Tokens are matched case-insensitively against the script source and
# are deliberately language-agnostic: ``activate`` covers both the
# AppleScript verb and JXA's ``.activate()``; ``keystroke`` / ``key
# code`` / ``click`` / ``perform action`` cover synthesized input in
# either dialect.
#
# We err toward over-matching: a false positive merely serializes a
# script that might have been safe to parallelize, while a false
# negative lets two agents collide — the exact bug the lease exists to
# prevent.  Pure reads (e.g. ``entire contents of window 1`` with no
# ``activate``) and background dictionary calls (``tell application
# "Mail" to …``) contain none of these tokens and stay fully parallel.
_FOCUS_STEALING_TOKENS: tuple[str, ...] = (
    "activate",
    "keystroke",
    "key code",
    "click",
    "perform action",
    "set frontmost",
    "set visible",
)


def script_needs_desktop(script: str) -> bool:
    """Return True when *script* would steal focus or synthesize input.

    Used to decide whether a ``run_osascript`` / ``run_osascript_file``
    call must hold the desktop lease.  Background scripting (app
    dictionary verbs, read-only Accessibility queries) returns False and
    is allowed to run concurrently with other agents.

    The match is a case-insensitive substring scan — AppleScript and JXA
    are not worth parsing for this, and the cost of an occasional false
    positive (an extra serialized script) is far lower than the cost of a
    false negative (two agents fighting over the foreground).
    """
    if not script:
        return False
    haystack = script.lower()
    return any(tok in haystack for tok in _FOCUS_STEALING_TOKENS)


# ``POSIX file "<path>"`` is AppleScript's canonical way to name a host
# file (used by ``make new attachment``, ``read``, ``open for access``, …).
# We rewrite ONLY these path literals so virtual session paths point at the
# real sandbox — and so free-text fields like an email ``content`` / ``subject``
# that merely mention "/output/…" are never mangled.  The capture groups keep
# the surrounding ``POSIX file "`` / closing ``"`` intact so only the path
# itself is replaced.  Whitespace around the path token is tolerated.
_POSIX_FILE_RE = re.compile(r'(POSIX file\s+")([^"]+)(")')


def rewrite_virtual_posix_paths(
    script: str,
    session_files: Optional[str] = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Remap virtual session paths inside ``POSIX file "…"`` literals.

    Inline AppleScript runs on the *host*, so a ``POSIX file "/output/foo"``
    resolves against the host filesystem root — not the agent's session
    sandbox (``$SESSION_FILES``) where ``write_file`` actually saved the file.
    The result is a silent failure: ``make new attachment`` with a host-root
    path attaches nothing.  This mirrors :func:`resolve_app_path`-style
    resolution for paths *embedded in* a script, complementing the script
    *path* remap that ``run_osascript_file`` already does.

    Resolution per captured path (same order as the file-path resolver):

    1. Relative path (no leading ``/``) → leave unchanged.
    2. Existing host path (e.g. ``/Applications/…``, ``/Users/…``) → leave.
    3. Already under ``$SESSION_FILES`` → leave (already expanded).
    4. ``$SESSION_FILES<path>`` exists, or its parent dir exists (the file is
       about to be created) → remap to the real sandbox path.
    5. Otherwise leave unchanged so the caller surfaces a clear error.

    Args:
        script: AppleScript / JXA source.
        session_files: Sandbox root.  Defaults to ``$SESSION_FILES`` from the
            environment; an empty / unset value disables remapping entirely.

    Returns:
        ``(rewritten_script, rewrites)`` where ``rewrites`` is a list of
        ``(original, replacement)`` pairs (empty when nothing changed), so
        the caller can surface what was remapped to the model.
    """
    if not script:
        return script, []
    root = session_files if session_files is not None else os.environ.get(
        "SESSION_FILES", ""
    )
    if not root:
        return script, []
    session_root = Path(root)
    session_prefix = str(session_root)
    rewrites: list[tuple[str, str]] = []

    def _repl(match: "re.Match[str]") -> str:
        prefix, raw, suffix = match.group(1), match.group(2), match.group(3)
        if not raw.startswith("/"):
            return match.group(0)
        # Real host path (Safari.app, /Users/…, /System/…) — leave alone.
        if Path(raw).exists():
            return match.group(0)
        # Already an expanded sandbox path — don't double-prefix.
        if raw == session_prefix or raw.startswith(session_prefix + "/"):
            return match.group(0)
        candidate = session_root / raw.lstrip("/")
        # Remap when the target exists (read / attach) or its parent dir
        # exists (file about to be created), mirroring the file resolver.
        if candidate.exists() or candidate.parent.is_dir():
            rewrites.append((raw, str(candidate)))
            return f"{prefix}{candidate}{suffix}"
        return match.group(0)

    return _POSIX_FILE_RE.sub(_repl, script), rewrites
