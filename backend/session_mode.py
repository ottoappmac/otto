"""Per-session working mode: agent, ask, plan, or debug.

Ask and Plan are read-only. That is enforced in code:

* the session backend refuses ``write_file`` / ``edit_file`` / ``edit`` inside
  the project folder, for the orchestrator and every sub-agent (they share the
  backend). Ask also refuses those writes anywhere, including ``/output/``;
* ``execute`` runs with the project removed from the sandbox write paths;
* the action policy denies every category that changes something outside the
  project (mail, messages, calendar, settings, browser clicks, unreviewed
  tools, and so on).

The prompt text for Ask, Plan, and Debug is advisory. A small model may ignore it;
the guards above do not depend on it.

Scheduled and trigger runs ignore the mode and always run as ``agent``.
"""

from __future__ import annotations

MODES: tuple[str, ...] = ("agent", "ask", "plan", "debug")
DEFAULT_MODE = "agent"

# Plan and Debug need room to explore, so they never run as Quick.
DEEP_MODES = frozenset({"plan", "debug"})

# Marker the Debug prompt asks the agent to put on every temporary logging
# line. The Changes view counts leftovers so cleanup is checked, not trusted.
DEBUG_MARKER = "otto-debug"

ASK_ADDENDUM = (
    "<session_mode>\n"
    "You are in ASK mode. Answer questions about the code. You cannot change anything.\n"
    "- Explore with read-only tools (ls, glob, grep, read_file) and read-only "
    "shell commands. File writes and edits are refused, including under /output/.\n"
    "- Do not claim any change was made. Tell the user to switch to Agent mode "
    "to make changes.\n"
    "</session_mode>"
)

PLAN_ADDENDUM = (
    "<session_mode>\n"
    "You are in PLAN mode. Nothing in the project folder can be changed.\n"
    "- Explore with read-only tools (ls, glob, grep, read_file) and read-only "
    "shell commands. File writes and edits in the project are refused.\n"
    "- If the request is ambiguous, ask the user short questions in your reply "
    "before planning.\n"
    "- Write the plan as Markdown to /output/plan.md and show the same plan "
    "in your reply. Cover: goal, files to change (real paths you checked), "
    "ordered steps, how to verify, and risks.\n"
    "- Do not claim any change was made. The user can switch to Agent mode "
    "to carry the plan out.\n"
    "</session_mode>"
)

DEBUG_ADDENDUM = (
    "<session_mode>\n"
    "You are in DEBUG mode. Find the cause before you change anything.\n"
    "1. Reproduce the problem. Run the failing command or test and read the "
    "real output.\n"
    "2. List two or three hypotheses, most likely first.\n"
    "3. Test them with evidence. If you add temporary logging, put the "
    f"marker `{DEBUG_MARKER}` on the same line (for example "
    f"`print(...)  # {DEBUG_MARKER}`), run again, and read the output.\n"
    "4. Make the smallest fix that explains the evidence.\n"
    f"5. Remove every `{DEBUG_MARKER}` line, then re-run the original "
    "reproduction to prove it is fixed.\n"
    "6. If the project has tests, add a regression test that fails without "
    "the fix and passes with it, then run the suite for the code you touched.\n"
    "Say which hypothesis the evidence confirmed. Do not guess when you can "
    "measure.\n"
    "</session_mode>"
)


def normalize_mode(value: object) -> str:
    """Return a known mode, or the default for anything else."""
    text = str(value or "").strip().lower()
    return text if text in MODES else DEFAULT_MODE


def effective_mode(mode: object, *, unattended: bool = False) -> str:
    """The mode that applies to a run. Unattended runs are always ``agent``."""
    if unattended:
        return DEFAULT_MODE
    return normalize_mode(mode)


def addendum_for(mode: str) -> str | None:
    if mode == "ask":
        return ASK_ADDENDUM
    if mode == "plan":
        return PLAN_ADDENDUM
    if mode == "debug":
        return DEBUG_ADDENDUM
    return None


def count_debug_markers(text: str) -> int:
    """How many lines of *text* carry the debug marker."""
    return sum(1 for line in (text or "").splitlines() if DEBUG_MARKER in line)
