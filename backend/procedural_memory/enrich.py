"""LLM-written intents and lessons for an admitted run, with a deterministic fallback.

One model call per admitted run (never per retrieval). The model writes:
  * ``intent``: the task in generic terms, values stripped ("Numbers: export the open spreadsheet to CSV");
  * one intent per segment;
  * up to three lessons in ReasoningBank's shape (title, one sentence, short content), phrased as **checks** so a
    lesson from a mislabelled run costs one verification step instead of blocking a correct path.

The exact steps stay the deterministic compaction's; the model never rewrites them. Without a model (or when its
answer cannot be parsed) intents come from ``compact.fallback_intent`` and no lessons are written.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

from backend.procedural_memory import compact

logger = logging.getLogger(__name__)

LLMCall = Callable[[str], Awaitable[str]]
MAX_LESSONS = 3

PROMPT = """You write reusable notes about how a computer-use agent did a task on macOS or in a browser, for \
other agents doing *different* tasks in the same app later.

Task given to the agent:
{task}

Apps used: {apps}
Outcome: {outcome} (from {source}){outcome_note}

Steps, grouped into segments (values replaced by <slots>; "ok"/"FAILED" is what the tool returned):
{segments}

Facts observed (actions that changed nothing):
{facts}

Return JSON only, no prose:
{{"intent": "<App>: <what the task does, generic, no names/values/paths, <= 14 words>",
  "segments": [{{"ord": <n>, "intent": "<App>: <what this segment does, generic, <= 12 words>"}}, ...],
  "lessons": [{{"title": "<= 8 words", "sentence": "<one sentence>", "content": "<1-3 sentences>"}}]}}

Rules for lessons (at most {max_lessons}; [] if nothing general was learned):
- About the app or the tools, useful for other tasks: where a command is, what a control does, what failed and what \
worked instead.
- Phrase each as a check or a way to do something ("After X, confirm Y"; "Add an action by double-clicking it"), \
never as "never do X".
- {failure_rule}
- No file names, values, people, URLs or text from this task."""

SUCCESS_RULE = "This run succeeded: note what made it work that is not obvious."
FAILURE_RULE = ("This run failed. The cause may not be visible: say what to check, and what to try instead the next "
                "time so that the task succeeds (another tool, menu, script or order of steps), not what to avoid. "
                "Do not describe the failed step sequence as a procedure.")


def _segments_text(segments: list[dict[str, Any]]) -> str:
    lines = []
    for s in segments:
        lines.append(f"[{s['ord']}] app={s.get('app') or '?'} trace={s.get('label')}")
        for ln in s.get("all_lines", s.get("recipe", []))[:16]:
            lines.append(f"    {ln}")
    return "\n".join(lines) or "(none)"


def build_prompt(ep: dict[str, Any], outcome: str, source: str) -> str:
    facts = "\n".join(f"- {f['text']}" for f in ep.get("facts", [])) or "(none)"
    note = "" if outcome == "success" else " — the last segment is where it went wrong"
    return PROMPT.format(task=compact.strip_values(ep.get("task_text", ""))[:1200], apps=", ".join(ep.get("apps") or []),
                         outcome=outcome, source=source, outcome_note=note, segments=_segments_text(ep["segments"]),
                         facts=facts, max_lessons=MAX_LESSONS,
                         failure_rule=FAILURE_RULE if outcome == "failure" else SUCCESS_RULE)


def _parse(text: str) -> dict[str, Any] | None:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _clean(s: Any, limit: int) -> str:
    return " ".join(str(s or "").split())[:limit]


def fallback(ep: dict[str, Any]) -> dict[str, Any]:
    app = (ep.get("apps") or [None])[0]
    return {"intent": compact.fallback_intent(app, ep.get("task_text", "")),
            "segments": {s["ord"]: s.get("fallback_intent") or "" for s in ep["segments"]},
            "lessons": [], "source": "fallback"}


async def enrich(ep: dict[str, Any], outcome: str, source: str, llm: LLMCall | None) -> dict[str, Any]:
    base = fallback(ep)
    if llm is None:
        return base
    try:
        raw = await llm(build_prompt(ep, outcome, source))
    except Exception:  # noqa: BLE001 — enrichment is best-effort; the run is still admitted with fallbacks
        logger.debug("[procmem] enrichment call failed", exc_info=True)
        return base
    data = _parse(raw)
    if data is None:
        return base
    out = dict(base, source="llm")
    if data.get("intent"):
        out["intent"] = _clean(data["intent"], 160)
    segs = dict(base["segments"])
    for s in data.get("segments") or []:
        if isinstance(s, dict) and isinstance(s.get("ord"), int) and s.get("intent"):
            segs[s["ord"]] = _clean(s["intent"], 140)
    out["segments"] = segs
    lessons = []
    for les in (data.get("lessons") or [])[:MAX_LESSONS]:
        if not isinstance(les, dict) or not les.get("title") or not (les.get("sentence") or les.get("content")):
            continue
        lessons.append({"title": _clean(les["title"], 80), "sentence": _clean(les.get("sentence"), 240),
                        "content": _clean(les.get("content"), 480)})
    out["lessons"] = lessons
    return out


async def default_llm() -> LLMCall | None:
    """The memory model Otto already configures (``MemoryConfig.llm_family``), or None if it cannot be built."""
    try:
        from backend.config import AppConfig
        from backend.memory_relevance import _create_ranking_model
        cfg = await AppConfig.aload()
        cfg.apply_to_environ()
        model = _create_ranking_model(cfg.memory, cfg.llm.provider, mlx_max_tokens=1500)
    except Exception:  # noqa: BLE001
        logger.debug("[procmem] no enrichment model", exc_info=True)
        return None

    async def call(prompt: str) -> str:
        from langchain_core.messages import HumanMessage
        res = await model.ainvoke([HumanMessage(content=prompt)])
        content = getattr(res, "content", res)
        return content if isinstance(content, str) else json.dumps(content)

    return call
