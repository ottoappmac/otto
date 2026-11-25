"""Turn a persisted transcript into episodes: deterministic, no LLM.

Input is the session transcript (``{session_id}.messages.json``, one JSON row per line, or the list the sessions API
returns): rows of type ``user``, ``agent``, ``tool_call`` (``content`` = tool name, ``metadata.args``,
``metadata.tool_call_id``) and ``tool_result`` (``content`` = result text, ``metadata.tool_call_id``). Rows made by a
delegated agent carry ``metadata.subagent`` (``"browser-agent #2"``); the orchestrator's ``task`` call that started
it holds the instruction (``args.description``) and the agent (``args.subagent_type``). The n-th ``task`` call for
an agent type is that type's ``#n`` run.

Output per run: compacted steps, segments with their own evidence, script artifacts, and no-op facts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# Agents whose runs are learned from. ``macarena-desktop-agent`` is the benchmark copy of the desktop agent.
DESKTOP_AGENTS = frozenset({"macos-desktop-agent", "macarena-desktop-agent"})
BROWSER_AGENTS = frozenset({"browser-agent"})
LEARNING_AGENTS = DESKTOP_AGENTS | BROWSER_AGENTS


def family(agent: str) -> str:
    return "browser" if agent in BROWSER_AGENTS else "desktop"


# Tools that only look. Not part of a recipe; their results still feed the index -> label map.
READ_TOOLS = frozenset({
    "get_screen_controls", "read_screen", "read_screen_vision", "observe", "read_app_dom", "list_apps",
    "find_text_on_screen", "get_control_value", "get_control_values", "read_dialog", "read_table", "menu_search",
    "app_guide", "wait_for_controls", "wait_until", "capture_app_screenshot", "dump_ax_tree",
    "inspect_app_dictionary", "query_mail_store",
    "browser_snapshot", "browser_take_screenshot", "browser_console_messages", "browser_network_requests",
    "browser_tabs", "browser_wait_for",
})
# Tools that are neither reads nor actions on an app.
META_TOOLS = frozenset({"write_todos", "recall_procedure", "task", "think", "grep_large_results", "read_file",
                        "ls", "glob", "grep", "web_research"})
END_TOOLS = frozenset({"vm_done", "vm_fail"})
SCRIPT_TOOLS = frozenset({"run_osascript", "browser_run_code", "browser_evaluate"})

# Argument keys that do not carry over to another run: screen points and per-launch element ids.
_COORD_KEYS = frozenset({"x", "y", "x2", "y2", "to_x", "to_y", "from_x", "from_y", "start_x", "start_y",
                         "end_x", "end_y", "coordinate", "coordinates", "ref", "element_ref"})
_INDEX_KEYS = ("index", "from_index", "to_index", "element", "control")
# Argument keys whose value is something the user typed: replaced by a slot.
_TEXT_KEYS = frozenset({"text", "value", "values", "keys_text", "content"})

# Result text patterns.
_ERROR_RE = re.compile(
    r'("ok":\s*false|^\s*error\b|^blocked:|not typed:|no controls match|was called \d+ times with identical|'
    r'\[check: |no turns left|not run yet|execution error|could not|failed to|unknown tool)', re.I | re.M)
_NOOP_RE = re.compile(r"same \d+ controls as the last read", re.I)
_CHANGE_RE = re.compile(
    r"(controls added or gone|controls now|dialog now|now open:|window '|the field now reads|went to '|"
    r"chose .+ in |typed '|\"ok\":\s*true)", re.I)
_PRESSED_RE = re.compile(r"#(\d+)\s*\((\w+)\s+'([^']*)'\)")
_LISTED_RE = re.compile(r"\[(\d+)\]([A-Za-z]{1,4})'([^']{0,80})'")
_APP_ARG_KEYS = ("app_name", "app", "application", "bundle_id")
_TELL_RE = re.compile(r'tell application "([^"]+)"', re.I)
_SAVE_RE = re.compile(r"\b(save|export|save as|duplicate|print)\b", re.I)

MAX_ARTIFACT_BYTES = 4096


@dataclass
class Step:
    i: int                      # position in the run's call list
    tool: str
    args: dict[str, Any]        # transferable arguments only
    raw_args: dict[str, Any]
    result: str                 # one status line
    kind: str                   # read | act | script | end | meta
    ok: bool
    noop: bool = False
    changed: bool = False       # the result shows the screen or state changed
    app: str | None = None
    target: str | None = None   # control label resolved from an index
    keep: bool = True           # part of the recipe

    def line(self) -> str:
        """One recipe line, values replaced by slots."""
        args = dict(self.args)
        parts = []
        if self.target:
            parts.append(repr(self.target))
        for k, v in args.items():
            if k in _APP_ARG_KEYS and self.tool not in ("launch_app", "activate_app", "open_app"):
                continue
            parts.append(f"{k}={v!r}" if not isinstance(v, str) or len(v) < 80 else f"{k}=<{k}>")
        return f"{self.tool}({', '.join(parts)})"

    def as_dict(self) -> dict[str, Any]:
        return {"i": self.i, "tool": self.tool, "args": self.args, "result": self.result, "kind": self.kind,
                "ok": self.ok, "noop": self.noop, "changed": self.changed, "app": self.app,
                "target": self.target, "keep": self.keep}


@dataclass
class Segment:
    ord: int
    step_from: int
    step_to: int
    app: str | None
    steps: list[Step]
    label: str = "unknown"       # success | failure | unknown, from the trace's own evidence
    evidence: str = ""
    intent: str = ""

    def recipe(self, max_lines: int = 12) -> list[str]:
        lines = [s.line() for s in self.steps if s.keep and s.kind in ("act", "script")]
        return lines[:max_lines]


@dataclass
class Fact:
    app: str | None
    text: str
    source: str                   # noop_detector | result_mismatch


@dataclass
class Artifact:
    kind: str                     # applescript | shell | jxa | playwright
    body: str
    segment_ord: int
    writes_or_sends: bool


@dataclass
class Run:
    """One delegated agent run inside a session."""
    session_id: str
    subagent_tag: str             # "browser-agent #2", or the agent name for a direct-agent session
    agent: str
    task_text: str
    final_text: str
    calls: list[dict[str, Any]] = field(default_factory=list)   # tool_call rows with their results

    @property
    def family(self) -> str:
        return family(self.agent)


@dataclass
class Episode:
    run: Run
    steps: list[Step]
    segments: list[Segment]
    facts: list[Fact]
    artifacts: list[Artifact]
    apps: list[str]
    self_report: str              # success | failure | unknown, from the agent's own finish
    infra_error: bool

    @property
    def actions(self) -> int:
        return sum(1 for s in self.steps if s.kind in ("act", "script"))


# ── reading transcripts ─────────────────────────────────────────────────────────────────────────────────────

def load_rows(source: str | bytes | Iterable[dict]) -> list[dict[str, Any]]:
    """Rows from a JSONL text, a JSON list text, or an iterable of dicts."""
    if isinstance(source, (str, bytes)):
        text = source.decode() if isinstance(source, bytes) else source
        stripped = text.lstrip()
        if stripped.startswith("["):
            data = json.loads(stripped)
            return [r for r in data if isinstance(r, dict)]
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict):
                    rows.append(r)
        return rows
    return [r for r in source if isinstance(r, dict)]


def _meta(row: dict) -> dict:
    m = row.get("metadata")
    return m if isinstance(m, dict) else {}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content)
    return "" if content is None else str(content)


def split_runs(rows: list[dict[str, Any]], session_id: str, direct_agent: str | None = None) -> list[Run]:
    """The learning-agent runs in one session.

    ``direct_agent``: the session's own agent when it ran one directly (benchmark sessions, "agent" mode). Its
    untagged rows are one run whose instruction is the first user message.
    """
    runs: dict[str, Run] = {}
    order: list[str] = []
    counters: dict[str, int] = {}
    pending_results: dict[str, dict] = {}
    task_calls: dict[str, tuple[str, str]] = {}      # tool_call_id -> (tag, agent)
    first_user = ""

    def _get(tag: str, agent: str, task_text: str = "") -> Run:
        if tag not in runs:
            runs[tag] = Run(session_id=session_id, subagent_tag=tag, agent=agent, task_text=task_text,
                            final_text="")
            order.append(tag)
        return runs[tag]

    for row in rows:
        rtype = row.get("type")
        meta = _meta(row)
        tag = meta.get("subagent")
        if rtype == "user" and not first_user:
            first_user = _text(row.get("content"))
        if rtype == "tool_call" and row.get("content") == "task" and not tag:
            args = meta.get("args") or {}
            agent = str(args.get("subagent_type") or "")
            counters[agent] = counters.get(agent, 0) + 1
            t = f"{agent} #{counters[agent]}"
            task_calls[str(meta.get("tool_call_id") or "")] = (t, agent)
            if agent in LEARNING_AGENTS:
                _get(t, agent, str(args.get("description") or ""))
            continue
        if rtype == "tool_result" and not tag and meta.get("name") == "task":
            t, agent = task_calls.get(str(meta.get("tool_call_id") or ""), ("", ""))
            if t in runs:
                runs[t].final_text = _text(row.get("content"))
            continue
        if tag:
            agent = str(tag).split(" #", 1)[0]
            if agent not in LEARNING_AGENTS:
                continue
            run = _get(str(tag), agent)
        elif direct_agent in LEARNING_AGENTS:
            run = _get(direct_agent, direct_agent)
        else:
            continue
        if rtype == "tool_call":
            call = {"tool": str(row.get("content") or meta.get("name") or ""),
                    "args": meta.get("args") if isinstance(meta.get("args"), dict) else {},
                    "id": str(meta.get("tool_call_id") or ""), "result": None, "ts": row.get("ts")}
            run.calls.append(call)
            if call["id"]:
                pending_results[call["id"]] = call
        elif rtype == "tool_result":
            cid = str(meta.get("tool_call_id") or "")
            call = pending_results.pop(cid, None)
            if call is None:
                for c in reversed(run.calls):
                    if c["result"] is None:
                        call = c
                        break
            if call is not None:
                call["result"] = _text(row.get("content"))
        elif rtype == "agent" and not tag and direct_agent in LEARNING_AGENTS:
            run.final_text = _text(row.get("content")) or run.final_text
        elif rtype == "agent" and tag:
            run.final_text = _text(row.get("content")) or run.final_text

    for run in runs.values():
        if not run.task_text and run.subagent_tag == direct_agent:
            run.task_text = instruction(first_user)
    return [runs[t] for t in order if runs[t].calls]


def instruction(text: str) -> str:
    """The task itself from a harness prompt ("Task: <instruction>\n\n<screen, budget, rules>"); other text
    unchanged."""
    t = (text or "").strip()
    if t.lower().startswith("task:"):
        t = t[5:].strip()
        return t.split("\n\n", 1)[0].strip()
    return t


# ── compaction ──────────────────────────────────────────────────────────────────────────────────────────────

def _kind(tool: str) -> str:
    if tool in END_TOOLS:
        return "end"
    if tool in READ_TOOLS:
        return "read"
    if tool in META_TOOLS:
        return "meta"
    if tool in SCRIPT_TOOLS:
        return "script"
    return "act"


_SHELL_WRITE_RE = re.compile(
    r"(?<![0-9&>])>{1,2}(?!\s*/dev/null)(?!&)|\btee\b|\b(cp|mv|ditto|ln|mkdir|touch|unzip|rm|chmod|open)\s|"
    r"\bsed\s+(-[a-zA-Z]*\s+)*-i|\b(pip3?|brew|npm|conda|gem)\s+install\b|\bpython[0-9.]*\b|"
    r"\bdefaults\s+write\b|\bplutil\s+-(replace|insert|remove|convert)\b|\bosascript\b|\bkillall\b")


def _read_only_shell(tool: str, args: dict) -> bool:
    """A ``run_osascript(shell_command=...)`` that only looks (cat, ls, test, grep): a read, not a recipe step."""
    if tool != "run_osascript" or not args.get("shell_command") or args.get("script"):
        return False
    return not _SHELL_WRITE_RE.search(str(args.get("shell_command")))


def _status_line(result: str | None) -> str:
    if result is None:
        return "no result"
    text = " ".join(result.split())
    return text[:160]


def _app_of(tool: str, args: dict, result: str) -> str | None:
    for k in _APP_ARG_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    if tool == "run_osascript":
        m = _TELL_RE.search(str(args.get("script") or ""))
        if m:
            return m.group(1)
    if tool.startswith("browser_"):
        url = str(args.get("url") or "")
        m = re.match(r"https?://([^/]+)", url)
        if m:
            return m.group(1).lower().removeprefix("www.")
    m = re.search(r"\(frontmost app: '([^']+)'\)", result or "")
    if m:
        return m.group(1)
    return None


def _labels_from(result: str | None, into: dict[int, str]) -> None:
    for m in _LISTED_RE.finditer(result or ""):
        into[int(m.group(1))] = m.group(3)
    for m in _PRESSED_RE.finditer(result or ""):
        into[int(m.group(1))] = m.group(3)


def _transferable(tool: str, args: dict, labels: dict[int, str], result: str = "") -> tuple[dict, str | None]:
    out: dict[str, Any] = {}
    own = _PRESSED_RE.search(result or "")        # "Pressed control #95 (Button 'Quick Action')"
    target = own.group(3) if own else None
    for k, v in args.items():
        if k in _COORD_KEYS:
            continue
        if k in _INDEX_KEYS:
            if isinstance(v, int) and v in labels:
                target = target or labels[v]
            continue
        if k in _TEXT_KEYS and isinstance(v, str):
            out[k] = f"<{k}>"
            continue
        if k in ("script", "shell_command", "code", "function"):
            out[k] = f"<{k}>"
            continue
        if k == "fields" and isinstance(v, dict):
            out[k] = {str(name): "<value>" for name in v}
            continue
        if k == "path" and tool != "menu_select" and isinstance(v, str):
            out[k] = strip_values(v) if "/" in v or "." in v else v
            continue
        if k == "steps" and tool == "batch_actions":
            out[k] = f"<{len(str(v).splitlines())} steps>"
            continue
        out[k] = v
    return out, target


def compact(run: Run) -> Episode:
    labels: dict[int, str] = {}
    steps: list[Step] = []
    current_app: str | None = None
    for i, call in enumerate(run.calls):
        tool, args, result = call["tool"], call.get("args") or {}, call.get("result")
        kind = _kind(tool)
        if kind == "script" and _read_only_shell(tool, args):
            kind = "read"
        app = _app_of(tool, args, result or "") or current_app
        if tool in ("launch_app", "activate_app", "open_app") and app:
            current_app = app
        elif app and kind in ("act", "script", "read"):
            current_app = app
        targs, target = _transferable(tool, args, labels, result or "")
        text = result or ""
        ok = result is not None and not _ERROR_RE.search(text[:400])
        noop = kind == "act" and bool(_NOOP_RE.search(text))
        changed = kind in ("act", "script") and ok and not noop and bool(_CHANGE_RE.search(text))
        if tool == "run_osascript" and ok:
            changed = True
        steps.append(Step(i=i, tool=tool, args=targs, raw_args=args, result=_status_line(result), kind=kind,
                          ok=ok, noop=noop, changed=changed, app=app, target=target))
        _labels_from(result, labels)

    _mark_kept(steps)
    segments = _segment(steps)
    facts = _facts(steps)
    artifacts = _artifacts(steps, segments)
    apps = []
    for s in steps:
        if s.app and s.app not in apps and s.kind != "meta":
            apps.append(s.app)
    infra = _infra_error(run)
    episode = Episode(run=run, steps=steps, segments=segments, facts=facts, artifacts=artifacts, apps=apps,
                      self_report=_self_report(run, steps), infra_error=infra)
    return episode


def _mark_kept(steps: list[Step]) -> None:
    """Recipe lines: actions and scripts that worked and changed something, without retries or repeats."""
    seen: set[str] = set()
    for idx, s in enumerate(steps):
        if s.kind not in ("act", "script"):
            s.keep = False
            continue
        if not s.ok or s.noop:
            s.keep = False
            continue
        key = s.line()
        nxt = steps[idx + 1] if idx + 1 < len(steps) else None
        # A step immediately redone the same way (the later one is the one that counts).
        if nxt is not None and nxt.tool == s.tool and nxt.line() == key and nxt.ok:
            s.keep = False
            continue
        if key in seen and s.kind == "act" and s.tool not in ("hotkey", "scroll", "press_key"):
            s.keep = False
            continue
        seen.add(key)


def _segment(steps: list[Step]) -> list[Segment]:
    """Split at app switches and after a save / export, and every 10 kept steps."""
    segments: list[Segment] = []
    cur: list[Step] = []
    cur_app: str | None = None

    def close() -> None:
        nonlocal cur
        if any(s.kind in ("act", "script") for s in cur):
            segments.append(Segment(ord=len(segments) + 1, step_from=cur[0].i, step_to=cur[-1].i, app=cur_app,
                                    steps=cur))
        cur = []

    for s in steps:
        if s.kind in ("meta", "end"):
            continue
        if cur and s.app and cur_app and s.app != cur_app and s.kind in ("act", "script"):
            close()
        if s.app and s.kind in ("act", "script"):
            cur_app = s.app
        cur.append(s)
        kept = sum(1 for x in cur if x.keep)
        is_save = s.kind == "act" and s.ok and (
            _SAVE_RE.search(str(s.raw_args.get("path") or "")) is not None
            or str(s.raw_args.get("keys") or "").lower() in ("command+s", "cmd+s", "command+shift+s"))
        if is_save or kept >= 10:
            close()
    close()
    for seg in segments:
        _label_segment(seg)
    return segments


def _label_segment(seg: Segment) -> None:
    """Evidence from the trace alone: the segment's last actions changed state and nothing failed after them."""
    acts = [s for s in seg.steps if s.kind in ("act", "script")]
    if not acts:
        return
    last_change = max((k for k, s in enumerate(acts) if s.changed), default=-1)
    failed_after = any((not s.ok or s.noop) for s in acts[last_change + 1:]) if last_change >= 0 else True
    if last_change >= 0 and not failed_after:
        seg.label = "success"
        seg.evidence = acts[last_change].result[:160]
    else:
        tail = acts[-1]
        seg.label = "failure"
        seg.evidence = tail.result[:160]


def _facts(steps: list[Step]) -> list[Fact]:
    """Observed facts: an action that changed nothing, tried again and again; typed text that did not take."""
    facts: list[Fact] = []
    by_key: dict[tuple, list[Step]] = {}
    for s in steps:
        if s.kind == "act" and s.noop:
            by_key.setdefault((s.app, s.tool, s.target or s.line()), []).append(s)
    for (app, tool, target), group in by_key.items():
        if len(group) >= 2:
            what = f"'{target}'" if target else target
            verb = {"press_control": "Pressing", "click": "Clicking", "double_click": "Double-clicking",
                    "click_text": "Clicking the text", "drag": "Dragging"}.get(tool, f"`{tool}` on")
            facts.append(Fact(app=app, source="noop_detector",
                              text=f"{verb} {what} changed nothing ({len(group)} tries, the controls stayed the same)."))
    for a, b in zip(steps, steps[1:]):
        if a.tool == "type_into_control" and b.tool == a.tool and a.target and a.target == b.target and \
                a.raw_args.get("text") == b.raw_args.get("text"):
            facts.append(Fact(app=a.app, source="result_mismatch",
                              text=f"type_into_control on '{a.target}' may add to its existing text; select all "
                                   "first and check the field reads only your text."))
            break
    return facts


def _artifacts(steps: list[Step], segments: list[Segment]) -> list[Artifact]:
    seg_of = {}
    for seg in segments:
        for s in seg.steps:
            seg_of[s.i] = seg.ord
    out: list[Artifact] = []
    for s in steps:
        if s.kind != "script" or not s.ok:
            continue
        if s.tool == "run_osascript":
            body = str(s.raw_args.get("script") or s.raw_args.get("shell_command") or "")
            kind = "shell" if s.raw_args.get("shell_command") else (
                "jxa" if str(s.raw_args.get("language") or "").lower().startswith("j") else "applescript")
        else:
            body = str(s.raw_args.get("code") or s.raw_args.get("function") or "")
            kind = "playwright"
        if not body.strip() or len(body.encode()) > MAX_ARTIFACT_BYTES:
            continue
        writes = bool(re.search(r"\b(rm|mv|cp|mkdir|touch|defaults write|send|delete|make new|save|>)\b|"
                                r"(?<![0-9&>])>{1,2}", body, re.I))
        out.append(Artifact(kind=kind, body=body, segment_ord=seg_of.get(s.i, 0), writes_or_sends=writes))
    # One copy of each body, the last one that worked.
    seen: dict[str, Artifact] = {}
    for a in out:
        seen[a.body] = a
    return list(seen.values())[-8:]


_INFRA_RE = re.compile(r"(cannot reach the benchmark vm|the vm tool server failed|closedresourceerror|"
                       r"connection (refused|reset)|rate limit|overloaded|timed out)", re.I)


def _infra_error(run: Run) -> bool:
    texts = [str(c.get("result") or "") for c in run.calls[-3:]] + [run.final_text]
    return any(_INFRA_RE.search(t) for t in texts)


def _self_report(run: Run, steps: list[Step]) -> str:
    for s in reversed(steps):
        if s.tool == "vm_done" and s.ok:
            return "success"
        if s.tool == "vm_fail":
            return "failure"
    text = (run.final_text or "").lower()
    if re.search(r"\b(could not|couldn't|unable to|failed|not possible|cannot)\b", text):
        return "failure"
    if text.strip():
        return "success"
    return "unknown"


# ── values out of text ──────────────────────────────────────────────────────────────────────────────────────

_QUOTED_RE = re.compile(r"\"[^\"]{1,80}\"|'[^']{1,80}'|“[^”]{1,80}”")
_PATH_RE = re.compile(r"(~|/)[\w./ -]+\.\w{1,5}\b|\b[\w-]+\.(txt|csv|xlsx?|docx?|pptx?|key|numbers|pages|pdf|png|"
                      r"jpe?g|md|json|ipynb|zip)\b", re.I)
_NUM_RE = re.compile(r"\b\d[\d,.:/-]*\b")
_URL_RE = re.compile(r"https?://\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")


def strip_values(text: str) -> str:
    """Task text with the specifics (quoted names, files, numbers, URLs, emails) replaced, for matching."""
    t = _URL_RE.sub("<url>", text or "")
    t = _EMAIL_RE.sub("<email>", t)
    t = _QUOTED_RE.sub("<name>", t)
    t = _PATH_RE.sub("<file>", t)
    t = _NUM_RE.sub("<n>", t)
    return " ".join(t.split())


def fallback_intent(app: str | None, text: str, max_words: int = 18) -> str:
    """An intent without an LLM: the app and the start of the value-stripped task text."""
    words = strip_values(text).split()
    body = " ".join(words[:max_words])
    return f"{app}: {body}" if app else body


def segment_fallback_intent(seg: Segment) -> str:
    targets = []
    for s in seg.steps:
        if s.keep:
            label = s.target or str(s.raw_args.get("path") or s.raw_args.get("keys") or "")
            label = label.strip()
            if label and label not in targets:
                targets.append(label)
    tools = []
    for s in seg.steps:
        if s.keep and s.tool not in tools:
            tools.append(s.tool)
    what = ", ".join(targets[:5]) or ", ".join(tools[:4])
    return f"{seg.app or 'app'}: {what}"
