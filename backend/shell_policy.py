"""Shell decisions that sit on top of the global action policy.

Read-only commands are auto-approved. A folder grant can name extra
command prefixes (``npm test*``) and can turn network on for its own
shell when the global switch is off. A host list, when set, is the
only outbound traffic that shell is given.
"""

from __future__ import annotations

import re

_META = re.compile(r"[;&|<>`]|\$\(")
_FIND_MUTATION = re.compile(r"(^|\s)(-delete|-exec|-ok)\b")

# A leading word (or ``git <subcommand>``) that does not write.
_READONLY = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "git blame",
    "git ls-files",
    "ls",
    "pwd",
    "cat",
    "head",
    "tail",
    "wc",
    "rg",
    "grep",
    "find",
)

# Long test suites. The ordinary shell timeout is too short for these.
_TEST_PREFIXES = (
    "pytest",
    "python -m pytest",
    "python3 -m pytest",
    "npm test",
    "npm run test",
    "pnpm test",
    "pnpm run test",
    "yarn test",
    "go test",
    "cargo test",
    "make test",
)

_TEST_TIMEOUT = 600


def _flat(command: str) -> str:
    return " ".join((command or "").strip().split())


def _has_metacharacters(command: str) -> bool:
    return bool(_META.search(command or ""))


def command_is_readonly(command: str) -> bool:
    """True when *command* is a single read-only invocation."""
    text = _flat(command)
    if not text or _has_metacharacters(text):
        return False
    if text.startswith("find ") and _FIND_MUTATION.search(text):
        return False
    return any(text == prefix or text.startswith(prefix + " ") for prefix in _READONLY)


def command_matches_allow(command: str, rules: list[str] | None) -> bool:
    """True when *command* matches a project allow rule.

    ``npm test`` matches that command and ``npm test -q``. ``npm test*``
    matches any command that starts with ``npm test``. Chained commands
    never match.
    """
    text = _flat(command)
    if not text or _has_metacharacters(text):
        return False
    for raw in rules or []:
        rule = " ".join(str(raw).strip().split())
        if not rule:
            continue
        if rule.endswith("*"):
            prefix = rule[:-1].rstrip()
            if prefix and text.startswith(prefix):
                return True
            continue
        if text == rule or text.startswith(rule + " "):
            return True
    return False


def allow_shell_command(
    command: str,
    *,
    labels: tuple[str, ...] | list[str] = (),
    rules: list[str] | None = None,
) -> str | None:
    """Why a shell command can skip the approval card, or None to ask.

    A danger label from the command screen always asks.
    """
    if labels:
        return None
    if command_is_readonly(command):
        return "read-only command"
    if command_matches_allow(command, rules):
        return "allowed for this project"
    return None


def timeout_for_command(command: str, default: int) -> int:
    """Give test commands a longer timeout than the ordinary shell default."""
    text = _flat(command)
    if any(text == prefix or text.startswith(prefix + " ") for prefix in _TEST_PREFIXES):
        return max(default, _TEST_TIMEOUT)
    return default


def shell_network(
    global_on: bool,
    project_on: bool,
    hosts: list[str] | None,
) -> tuple[bool, list[str]]:
    """``(allow_all, hosts)``.

    The global switch still opens network for every project. A project
    can open it for itself while the global switch stays off. A host
    list limits that traffic to those hosts.
    """
    cleaned = [item.strip() for item in (hosts or []) if item and str(item).strip()]
    enabled = bool(global_on or project_on)
    if not enabled:
        return False, []
    if cleaned:
        return False, cleaned
    return True, []
