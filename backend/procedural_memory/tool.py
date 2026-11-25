"""``recall_procedure``: the agent-facing tool.

Offered to ``macos-desktop-agent`` and ``browser-agent`` in the app when procedural memory is on, and to
``macarena-desktop-agent`` when the benchmark backend serves a snapshot. In the benchmark every call counts against
the step budget (the bridge counts it, ``scripts/macarena/bridge.py``), so a memory arm is compared with a no-memory
arm on the same budget; and each call's query, candidates and returned items are reported to the bridge, which puts
them in the run's results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.procedural_memory import compact, retrieve, service

logger = logging.getLogger(__name__)

TOOL_NAME = "recall_procedure"

DESCRIPTION = """Look up how a similar step was done before in this app: a short recipe (tool calls with <slots> for \
values), facts about the app (controls that do nothing, what worked instead) and checks. From *other* tasks: \
confirm each step against the screen and never copy values from it.

Describe the step generically, with the app: task="Numbers: export the open sheet to CSV", app="Numbers" \
(the app name, or the site for the browser). Call it before a multi-step part of the task you have not done in \
this app, at most once per part; put it in the same reply as your first look at the screen. If it says there is \
nothing similar, work it out and do not ask again for that part.

Follow-ups by id: detail="script" for the script a recipe used, detail="steps" for all its steps."""


_VENDORS = ("google ", "microsoft ", "apple ", "adobe ", "mozilla ")


def _short(name: str) -> str:
    low = " ".join(name.lower().split())
    for v in _VENDORS:
        if low.startswith(v) and len(low) - len(v) >= 4:
            return low[len(v):]
    return low


def resolve_app(app: str | None, known: list[str]) -> str | None:
    """The stored app name the agent meant: "Chrome" -> "Google Chrome", "LibreOffice Calc" -> "LibreOffice",
    "quicktime" -> "QuickTime Player". Exact (any case) first, then vendor-less, then the longest stored name the
    given one contains, then a stored name that contains it. ``app`` unchanged when nothing fits (so the search
    says there is nothing for that app, rather than searching another)."""
    if not app or not app.strip():
        return app
    given = " ".join(app.lower().split())
    by_low = {k.lower(): k for k in known}
    if given in by_low:
        return by_low[given]
    for k in known:
        if _short(k) == _short(given):
            return k
    inside = [k for k in known if len(k) >= 4 and k.lower() in given]
    if inside:
        return max(inside, key=len)
    around = [k for k in known if len(given) >= 4 and given in k.lower()]
    if len(around) == 1:
        return around[0]
    return app


async def _first_user_text(session_id: str | None) -> str:
    if not session_id:
        return ""
    try:
        from backend.config import get_app_data_dir
        path = get_app_data_dir() / "sessions" / f"{session_id}.messages.json"
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("type") == "user":
                    from backend.procedural_memory.compact import instruction
                    return instruction(str(row.get("content") or ""))
    except (OSError, ValueError):
        return ""
    return ""


async def run_recall(*, agent: str, session_id: str | None, task: str = "", app: str | None = None,
                     id: str | None = None, detail: str = "recipe",
                     current_task_text: str = "") -> tuple[str, dict[str, Any]]:
    """(text for the agent, a record of the call for logs and results)."""
    rt = service.runtime()
    st = service.serve_store(rt)
    record: dict[str, Any] = {"query": task, "app": app, "id": id, "detail": detail, "used": [], "candidates": [],
                              "abstained": False, "tokens": 0}
    if st is None:
        return "Procedure memory is off.", record
    built_with = st.get_meta().get("embedder")
    if built_with and built_with != service.get_embedder().name:
        record["error"] = f"snapshot embedder {built_with} != {service.get_embedder().name}"
        logger.warning("[procmem] %s; recall disabled", record["error"])
        return "Procedure memory is unavailable (embedding model mismatch).", record
    if id:
        text = retrieve.detail(st, id, detail if detail in ("recipe", "script", "steps") else "recipe")
        it = st.item(id)
        if it is not None:
            record["used"] = [{"item": it.id, "episode": it.episode_id, "kind": it.kind, "source_task": it.task_id,
                               "label_source": it.label_source, "detail": detail}]
        record["tokens"] = len(text) // 4
        return text, record
    if not task.strip():
        return "Give task: a generic description of the step, with the app.", record
    if app:
        resolved = resolve_app(app, st.apps(compact.family(agent)))
        if resolved != app:
            record["app_resolved"] = resolved
        app = resolved
    usage = st.usage_stats() if not rt.benchmark else {}
    # Benchmark: stored runs of the same task are left out (no reading its own answer), unless a smoke test asks
    # otherwise. App: never left out; a task the user asks for again is exactly what a saved run is for.
    if rt.benchmark and not rt.same_task:
        current = current_task_text or await _first_user_text(session_id)
    else:
        current = ""
    res = await retrieve.recall(st, service.get_embedder(), query=task, family=compact.family(agent), app=app,
                                floor=rt.floor, max_tokens=rt.max_tokens, current_task_text=current, usage=usage,
                                exact_only=rt.live)
    record.update(used=res.used(), candidates=[c.as_log() for c in res.candidates], abstained=res.abstained,
                  tokens=res.tokens, text=res.text)   # exactly what the agent got (memory_calls.jsonl)
    ws = service.write_store(rt)
    if ws is not None and session_id and res.returned:
        try:
            ws.record_usage([it.id for it in res.returned], session_id)
        except Exception:  # noqa: BLE001 — bookkeeping never breaks the agent
            logger.debug("[procmem] usage not recorded", exc_info=True)
    return res.text, record


PROMPT_HINT = """
## Procedure memory (recall_procedure)
You have `recall_procedure`: recipes and app facts from earlier runs of *other* tasks. Before a multi-step part of \
the task in an app (setting up a document, a dialog, a menu path you have not used here), call it once, in the same \
reply as your first look at the screen: task="<App>: <the step, generically>", app="<App>". Treat what it returns as \
hints: confirm each step against the screen, never copy values from it. If it says there is nothing similar, do not \
ask again for that part.
"""

# Step 0 of the benchmark prompt's numbered Workflow (the list the agent follows), only when the tool is bound.
STEP_ZERO = (
    "0. **Search memory first.** Your first reply calls `recall_procedure(task=\"<App>: <what you must do>\", "
    "app=\"<App>\")` once, with `observe()` in the same reply: the app is the one the task names (or the one on "
    "screen), and the step is described generically, with no names, numbers or file names from the task. Use what it "
    "returns as hints: it comes from other tasks, so confirm each step against the screen and never copy values from "
    "it. If it says there is nothing similar, carry on with step 1 and do not call it again for that part.\n"
)
_WORKFLOW_FIRST = "1. **Look first.**"


# The line that goes after the task in the first message of a run with the tool bound. Offline, on the benchmark's
# local model, step 0 in the system prompt alone got the first reply to call the tool about half the time; with
# this line right after the task, every time (scripts/procmem/README or the plan doc has the numbers).
FIRST_MESSAGE_NUDGE = (
    "Before anything else: your first reply calls recall_procedure(task=\"<App>: <what you must do, without names "
    "or values>\", app=\"<App>\") once, together with observe(). It returns steps and pitfalls from other tasks in "
    "that app."
)


def has_tool(tools: list[Any]) -> bool:
    return any(getattr(t, "name", None) == TOOL_NAME and (getattr(t, "metadata", None) or {}).get("procedure_memory")
               for t in tools)


def apply_prompt(system_prompt: str, tools: list[Any], *, agent: str | None = None) -> str:
    """``system_prompt`` with the memory instruction added when the tool is among ``tools``, else unchanged (an agent
    without memory gets exactly the prompt it had). Where the prompt has a numbered Workflow starting with "1. Look
    first." (the benchmark agent) the instruction becomes its step 0; elsewhere it is a section at the end."""
    if not has_tool(tools):
        return system_prompt
    if _WORKFLOW_FIRST in system_prompt:
        out, where = system_prompt.replace(_WORKFLOW_FIRST, STEP_ZERO + _WORKFLOW_FIRST, 1), "workflow step 0"
    else:
        out, where = system_prompt + PROMPT_HINT, "appended section"
    # Otto's INFO logs do not reach the benchmark backend's log file, and this line is what shows the instruction
    # got into the prompt, so it is a warning there.
    try:
        bench = service.runtime().benchmark
    except Exception:  # noqa: BLE001
        bench = False
    (logger.warning if bench else logger.info)(
        "[procmem] recall_procedure bound for %s; instruction added as %s (+%d chars)", agent or "?", where,
        len(out) - len(system_prompt))
    return out


def first_message(agent_name: str | None, query: str) -> str:
    """The model's copy of a run's first message: ``query`` plus the memory instruction when this agent is offered
    the tool, else ``query`` unchanged. Only the model sees the addition; the transcript keeps the user's text."""
    if not query.strip() or agent_name not in compact.LEARNING_AGENTS:
        return query
    try:
        if not service.runtime().serve_tool:
            return query
    except Exception:  # noqa: BLE001
        return query
    return query + "\n\n" + FIRST_MESSAGE_NUDGE


def prompt_hint(tools: list[Any]) -> str:
    """Kept for callers that only append: the section for the tool, or "" when it is not bound."""
    return PROMPT_HINT if has_tool(tools) else ""


def build_procedure_tools(agent_name: str | None, session_id: str | None) -> list[Any]:
    """The tool for this agent, or [] when memory is off or the agent does not learn."""
    if agent_name not in compact.LEARNING_AGENTS:
        return []
    try:
        rt = service.runtime()
    except ValueError:
        logger.warning("[procmem] bad benchmark memory settings; no recall tool", exc_info=True)
        return []
    if not rt.serve_tool:
        return []
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class RecallArgs(BaseModel):
        task: str = Field("", description="The step, generically, with the app: 'Numbers: export the open sheet to CSV'")
        app: str | None = Field(None, description="App name (or the site for the browser) to filter by")
        id: str | None = Field(None, description="An item id from an earlier answer, for its details")
        detail: str = Field("recipe", description="With id: 'recipe', 'steps' or 'script'")

    async def recall_procedure(task: str = "", app: str | None = None, id: str | None = None,
                               detail: str = "recipe") -> str:
        text, record = await run_recall(agent=agent_name, session_id=session_id, task=task, app=app, id=id,
                                        detail=detail)
        if rt.benchmark:
            from backend import macarena_native
            await macarena_native.report_memory(record)
        return text

    return [StructuredTool(name=TOOL_NAME, description=DESCRIPTION, args_schema=RecallArgs,
                           coroutine=recall_procedure, metadata={"procedure_memory": True})]
