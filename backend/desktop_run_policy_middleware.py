"""Run policies for the macOS desktop agents: recover from tool calls written as text, show the screen at the start,
and point out a tool that keeps failing.

Applies to ``macos-desktop-agent`` (the app's agent) and ``macarena-desktop-agent`` (the Otto-native benchmark
agent). Every other agent is untouched.

* **Tool call written as text.** Local models (Qwen through omlx) sometimes put a call in the reply text
  (``<tool_call>{"name": ...}</tool_call>`` or ``<function=name><parameter=x>...``) instead of the tool-calling
  interface. A reply with no tool calls ends the agent's run, so one such slip ended whole benchmark tasks. When the
  text parses as a call to a tool the agent has, it is turned into that call; when it does not, the model is asked
  once more, with a note saying why.
* **Benchmark: a first reply with no tool call.** The benchmark agent cannot ask anyone anything, so a first reply
  without a tool call (a plan, a question) would end the task unattempted. It is asked once more.
* **The screen at the start.** Before the first model call for a new request, ``observe`` is read and added after
  the request, so the agent starts knowing which app, window, file and dialog are in front instead of guessing.
* **The same tool failing.** After the same tool failed twice in a row, its result says so and names the tools that
  usually work instead (menus and dialogs through ``menu_select`` / ``read_dialog`` rather than another script).

Retries go through the rest of the middleware stack, so the benchmark bridge records each model call.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

APP_AGENT = "macos-desktop-agent"
BENCH_AGENT = "macarena-desktop-agent"
AGENTS = frozenset({APP_AGENT, BENCH_AGENT})

FAIL_STREAK = 2
START_TAG = "[screen when this request arrived, read by observe()]"

TEXT_CALL_NOTE = (
    "Your last reply wrote a tool call as text, so nothing ran. Call tools only through the tool-calling interface "
    "(no <tool_call> or <function=...> text in the reply)."
)
FIRST_TURN_NOTE = (
    "Nothing has been done yet and no one can answer questions here. Start working on the task with a tool call: "
    "observe() shows what is on screen. If the task cannot be done, call vm_fail."
)

# ── Tool calls written as text ─────────────────────────────────────────────

_HERMES = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_XML_FN = re.compile(r"<function=([\w.\-]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM = re.compile(r"<parameter=([\w.\-]+)>\s*(.*?)\s*(?:</parameter>|(?=<parameter=)|$)", re.S)
_MARKERS = ("<tool_call>", "<function=", "</tool_call>")


def _param_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def parse_text_calls(text: str) -> list[dict] | None:
    """The tool calls written in ``text`` as ``[{"name", "args"}]``; [] when there are none; None when the text
    has call markup that does not parse."""
    if not any(m in text for m in _MARKERS):
        return []
    calls: list[dict] = []
    for m in _HERMES.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except ValueError:
            return None
        name = obj.get("name") if isinstance(obj, dict) else None
        args = obj.get("arguments", obj.get("parameters", {})) if isinstance(obj, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                return None
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append({"name": name, "args": args})
    if not calls:
        for m in _XML_FN.finditer(text):
            args = {k: _param_value(v) for k, v in _XML_PARAM.findall(m.group(2))}
            calls.append({"name": m.group(1), "args": args})
    return calls or None


def _text_of(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def _tool_name(t: Any) -> str:
    return t.get("name", "") if isinstance(t, dict) else str(getattr(t, "name", "") or "")


def _last_ai(resp: ModelResponse) -> AIMessage | None:
    for m in reversed(resp.result):
        if isinstance(m, AIMessage):
            return m
    return None


# ── Failure streaks ────────────────────────────────────────────────────────

_FAIL_TEXT = re.compile(
    r"^(error|failed|menu not chosen|no such|could not|cannot|not in registry|control #\d+ not)|"
    r"['\"]ok['\"]\s*:\s*false|✗|execution error|syntax error", re.I | re.M)


def failed(msg: ToolMessage) -> bool:
    if getattr(msg, "status", None) == "error":
        return True
    return bool(_FAIL_TEXT.search(_text_of(msg)[:600]))


def fail_streak(messages: list[BaseMessage], name: str) -> int:
    """How many of the latest results of tool ``name`` failed in a row, looking back over the latest tool results
    (results of other tools in between end the streak)."""
    n = 0
    for m in reversed(messages):
        if not isinstance(m, ToolMessage):
            if isinstance(m, HumanMessage):
                break
            continue
        if m.name != name:
            break
        if not failed(m):
            break
        n += 1
    return n


def streak_hint(name: str, streak: int) -> str:
    if name in ("run_osascript", "run_osascript_file", "dump_ax_tree"):
        alt = ("If the script drives menus or dialogs, use menu_select / menu_search and read_dialog / fill_dialog "
               "instead; app_guide(app) says whether the app can be scripted at all.")
    elif name in ("press_control", "click", "type_into_control", "get_control_value"):
        alt = ("Read the screen again (observe(), get_screen_controls or read_dialog) before using an index; menu "
               "commands are more reliable through menu_select.")
    elif name == "menu_select":
        alt = "Find the item's real path with menu_search(app, 'word'), or check app_guide(app)."
    else:
        alt = "Change the approach instead of repeating the call: observe() first, then a different tool."
    return f"\n\n[harness] {name} has now failed {streak} times in a row. Do not retry it as is. {alt}"


# ── The middleware ─────────────────────────────────────────────────────────

class DesktopRunPolicyMiddleware(AgentMiddleware):
    def __init__(self, agent_name: str | None):
        super().__init__()
        self.agent_name = agent_name
        # Text of the start-of-request observe(), by request (the last human message's id or content).
        self._starts: dict[str, str] = {}

    # which agent --------------------------------------------------------

    def _agent(self) -> str | None:
        if self.agent_name in AGENTS:
            return self.agent_name
        from backend.action_policy.runtime import current_scope
        scope = current_scope()
        name = getattr(scope, "agent_name", None) if scope is not None else None
        return name if name in AGENTS else None

    # start state ----------------------------------------------------------

    @staticmethod
    def _request_key(messages: list[BaseMessage]) -> tuple[int, str] | None:
        """(position, key) of the last human message, when nothing has answered it yet."""
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if isinstance(m, HumanMessage) and not _is_start(m) and not _text_of(m).startswith("[harness]"):
                return i, m.id or str(hash(_text_of(m)))
        return None

    def _with_start(self, request: ModelRequest, observe: Callable[[Any], str | None]) -> ModelRequest:
        msgs = list(request.messages)
        found = self._request_key(msgs)
        tool = next((t for t in request.tools if _tool_name(t) == "observe" and not isinstance(t, dict)), None)
        if found is None or tool is None:
            return request
        i, key = found
        text = self._starts.get(key)
        if text is None:
            if any(isinstance(x, AIMessage) for x in msgs[i + 1:]):
                return request          # the request was answered before this middleware saw it
            text = observe(tool) or ""
            self._starts[key] = text
            if len(self._starts) > 64:
                self._starts.pop(next(iter(self._starts)))
        if not text:
            return request
        msgs.insert(i + 1, HumanMessage(content=f"{START_TAG}\n{text}"))
        return request.override(messages=msgs)

    @staticmethod
    def _observe_sync(tool: Any) -> str | None:
        try:
            return str(tool.invoke({}))
        except Exception:  # noqa: BLE001 — the start state is an extra; never block the run
            logger.debug("start-state observe failed", exc_info=True)
            return None

    @staticmethod
    async def _observe_async(tool: Any) -> str | None:
        try:
            return str(await tool.ainvoke({}))
        except Exception:  # noqa: BLE001
            logger.debug("start-state observe failed", exc_info=True)
            return None

    # malformed replies ----------------------------------------------------

    def _review(self, request: ModelRequest, resp: ModelResponse, agent: str) -> tuple[ModelResponse | None, str]:
        """(the response to use instead, or None) and the note for a retry ("" = no retry)."""
        ai = _last_ai(resp)
        if ai is None or ai.tool_calls:
            return None, ""
        text = _text_of(ai)
        calls = parse_text_calls(text)
        names = {_tool_name(t) for t in request.tools}
        if calls:
            if all(c["name"] in names for c in calls):
                fixed = AIMessage(
                    content=_strip_calls(text), id=ai.id,
                    tool_calls=[{"name": c["name"], "args": c["args"], "id": f"call_{uuid.uuid4().hex[:24]}",
                                 "type": "tool_call"} for c in calls],
                    response_metadata=ai.response_metadata, usage_metadata=ai.usage_metadata)
                logger.info("desktop policy: turned %d text tool call(s) into calls", len(calls))
                return ModelResponse(result=[fixed if m is ai else m for m in resp.result],
                                     structured_response=resp.structured_response), ""
            return None, TEXT_CALL_NOTE
        if calls is None:
            return None, TEXT_CALL_NOTE
        if agent == BENCH_AGENT and not any(isinstance(m, AIMessage) for m in request.messages):
            return None, FIRST_TURN_NOTE
        return None, ""

    @staticmethod
    def _retry_request(request: ModelRequest, resp: ModelResponse, note: str) -> ModelRequest:
        ai = _last_ai(resp)
        msgs = list(request.messages)
        if ai is not None:
            msgs.append(AIMessage(content=_text_of(ai) or "(no reply)"))
        msgs.append(HumanMessage(content=f"[harness] {note}"))
        return request.override(messages=msgs)

    # hooks ----------------------------------------------------------------

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]):
        agent = self._agent()
        if agent is None:
            return handler(request)
        request = self._with_start(request, self._observe_sync)
        resp = handler(request)
        fixed, note = self._review(request, resp, agent)
        if fixed is not None:
            return fixed
        if note:
            resp2 = handler(self._retry_request(request, resp, note))
            fixed2, _ = self._review(request, resp2, APP_AGENT)
            return fixed2 or resp2
        return resp

    async def awrap_model_call(self, request: ModelRequest,
                               handler: Callable[[ModelRequest], Awaitable[ModelResponse]]):
        agent = self._agent()
        if agent is None:
            return await handler(request)
        tool = next((t for t in request.tools if _tool_name(t) == "observe" and not isinstance(t, dict)), None)
        found = self._request_key(list(request.messages))
        if tool is not None and found is not None and found[1] not in self._starts \
                and not any(isinstance(x, AIMessage) for x in request.messages[found[0] + 1:]):
            self._starts[found[1]] = await self._observe_async(tool) or ""
        request = self._with_start(request, lambda _t: "")
        resp = await handler(request)
        fixed, note = self._review(request, resp, agent)
        if fixed is not None:
            return fixed
        if note:
            resp2 = await handler(self._retry_request(request, resp, note))
            fixed2, _ = self._review(request, resp2, APP_AGENT)
            return fixed2 or resp2
        return resp

    def _after_tool(self, request: ToolCallRequest, result: Any) -> Any:
        if not isinstance(result, ToolMessage) or not failed(result):
            return result
        name = request.tool_call.get("name") or ""
        state = request.state if isinstance(request.state, dict) else {}
        prior = fail_streak(list(state.get("messages") or []), name)
        streak = prior + 1
        if streak < FAIL_STREAK:
            return result
        if isinstance(result.content, str):
            result.content = result.content + streak_hint(name, streak)
        elif isinstance(result.content, list):
            result.content = [*result.content, {"type": "text", "text": streak_hint(name, streak)}]
        return result

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]):
        if self._agent() is None:
            return handler(request)
        return self._after_tool(request, handler(request))

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]):
        if self._agent() is None:
            return await handler(request)
        return self._after_tool(request, await handler(request))


def _is_start(m: BaseMessage) -> bool:
    return isinstance(m, HumanMessage) and _text_of(m).startswith(START_TAG)


def _strip_calls(text: str) -> str:
    text = _HERMES.sub("", text)
    text = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text, flags=re.S)
    text = _XML_FN.sub("", text)
    return text.strip()
