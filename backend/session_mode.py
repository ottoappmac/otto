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
the guards above do not depend on it. When a Plan turn ends in a plan, Otto
writes that reply to ``output/plan.md`` (created the first time, replaced
when the plan is revised). A question or a short note does not become the file.

Scheduled and trigger runs ignore the mode and always run as ``agent``.
"""

from __future__ import annotations

from pathlib import Path

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
    "before planning. A question is not a plan.\n"
    "- If /output/plan.md already exists, read it and revise that plan. "
    "Do not start over unless the user asked for a different plan.\n"
    "- Your final reply is the plan, in Markdown. Otto saves that reply to "
    "/output/plan.md: it creates the file the first time and replaces it "
    "when you revise the plan. Reply with the full plan, not a summary of "
    "what changed. Cover: goal, files to change (real paths you checked), "
    "ordered steps, how to verify, and risks.\n"
    "- Do not write /output/plan.md yourself, and do not claim any project "
    "change was made. The user can switch to Agent mode to carry the plan out.\n"
    "</session_mode>"
)

# Session file the plan reply is saved to. Not part of the project.
PLAN_FILE = "output/plan.md"

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


def plan_markdown_path(files_dir: Path) -> Path:
    """Absolute path of the session plan file."""
    return Path(files_dir) / PLAN_FILE


def final_assistant_text(messages: list) -> str:
    """Text of this turn's last assistant message that did not call tools.

    Messages before the latest user message belong to an earlier turn. Using
    them would write the previous plan back over a file this turn updated.
    """
    from langchain_core.messages import AIMessage, HumanMessage

    current: list = []
    for msg in messages or []:
        if isinstance(msg, HumanMessage):
            current = []
        else:
            current.append(msg)
    for msg in reversed(current):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "tool_calls", None):
            continue
        return _message_text(getattr(msg, "content", ""))
    return ""


def store_plan_markdown(files_dir: Path, reply: str) -> bool:
    """Create or replace ``output/plan.md`` from a plan-mode reply.

    Returns True when the file contents change. A short question or status
    line does not replace a plan already on disk, and does not create one.
    """
    text = _unwrap_markdown_fence(reply or "")
    path = plan_markdown_path(files_dir)
    existing = ""
    if path.is_file():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""
    if not _looks_like_plan(text):
        return False
    body = text if text.endswith("\n") else text + "\n"
    if existing == body:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def _looks_like_plan(text: str) -> bool:
    reply = text.strip()
    if _is_question(reply):
        return False
    if reply.lstrip().startswith("#"):
        return True
    return len(reply) >= 80 and "\n" in reply


def _is_question(text: str) -> bool:
    body = text.strip()
    return body.endswith("?") and "\n" not in body


def _unwrap_markdown_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip().startswith("```"):
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return ""
