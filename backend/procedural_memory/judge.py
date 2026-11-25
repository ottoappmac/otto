"""A trajectory judge for labelling runs, and the statistics Phase 0 uses to decide whether to trust it.

Built from what raised judge precision in the literature: the agent's final message is **excluded** (agents claim
success on most failed runs), the task is first split into key points, the verdict is checked against tool results
(the state the run actually read back), and several samples must agree unanimously (ZeroGUI: higher precision, lower
recall; false successes hurt memory more than missed ones).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

from backend.procedural_memory import compact
from backend.procedural_memory.enrich import LLMCall

PROMPT = """You check whether a computer-use agent really completed a task on macOS or in a browser. You see the \
task and the agent's tool calls with what each returned. You do NOT see the agent's own summary: agents often say \
they succeeded when they did not. Judge only from what the tool results show about the final state.

Task:
{task}

Tool calls and results (oldest first; long results cut):
{calls}

1. List the key points the task requires (each a checkable fact about the end state: a file exists with a name, a \
setting is on, a cell holds a value, a page is open).
2. For each key point, say whether a tool result shows it true at the end, shows it false, or does not show it.
3. Verdict: "success" only if every key point is shown true by a result after the last change; "failure" if any is \
shown false or the run ended in errors; "unknown" if the results do not show it either way.

Return JSON only: {{"key_points": [{{"point": "...", "status": "true|false|not_shown"}}], "verdict": \
"success|failure|unknown"}}"""

MAX_CALL_LINES = 120


def _calls_text(run: compact.Run) -> str:
    lines = []
    for c in run.calls[-MAX_CALL_LINES:]:
        if c["tool"] in compact.META_TOOLS:
            continue
        args = json.dumps(c.get("args") or {}, ensure_ascii=False)[:240]
        res = " ".join(str(c.get("result") or "").split())[:300]
        lines.append(f"- {c['tool']}({args}) -> {res}")
    return "\n".join(lines) or "(no calls)"


def build_prompt(run: compact.Run) -> str:
    return PROMPT.format(task=run.task_text[:2000], calls=_calls_text(run))


def _verdict(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("verdict")
            if v in ("success", "failure", "unknown"):
                return v
        except ValueError:
            pass
    m = re.search(r'"verdict"\s*:\s*"(success|failure|unknown)"', text)
    return m.group(1) if m else "unknown"


async def judge(run: compact.Run, llm: LLMCall, samples: int = 3) -> dict[str, Any]:
    """Unanimous vote over ``samples`` calls: success / failure only when every sample agrees."""
    votes = []
    prompt = build_prompt(run)
    for _ in range(max(1, samples)):
        try:
            votes.append(_verdict(await llm(prompt)))
        except Exception:  # noqa: BLE001
            votes.append("unknown")
    verdict = votes[0] if votes and all(v == votes[0] for v in votes) else "unknown"
    return {"verdict": verdict, "votes": votes}


# ── statistics ──────────────────────────────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


@dataclass
class Agreement:
    n: int
    said_success: int
    true_success_of_said: int
    said_failure: int
    true_failure_of_said: int
    actual_success: int
    actual_failure: int
    abstained: int

    def as_dict(self) -> dict[str, Any]:
        sp = self.true_success_of_said / self.said_success if self.said_success else None
        fp = self.true_failure_of_said / self.said_failure if self.said_failure else None
        return {
            "n": self.n, "abstained": self.abstained,
            "success_precision": sp, "success_precision_ci95": wilson(self.true_success_of_said, self.said_success),
            "success_recall": self.true_success_of_said / self.actual_success if self.actual_success else None,
            "failure_precision": fp, "failure_precision_ci95": wilson(self.true_failure_of_said, self.said_failure),
            "failure_recall": self.true_failure_of_said / self.actual_failure if self.actual_failure else None,
            "said_success": self.said_success, "said_failure": self.said_failure,
        }


def agreement(pairs: list[tuple[str, str]]) -> Agreement:
    """``pairs``: (predicted, actual) with actual in success | failure."""
    a = Agreement(n=len(pairs), said_success=0, true_success_of_said=0, said_failure=0, true_failure_of_said=0,
                  actual_success=sum(1 for _, t in pairs if t == "success"),
                  actual_failure=sum(1 for _, t in pairs if t == "failure"), abstained=0)
    for pred, truth in pairs:
        if pred == "success":
            a.said_success += 1
            a.true_success_of_said += truth == "success"
        elif pred == "failure":
            a.said_failure += 1
            a.true_failure_of_said += truth == "failure"
        else:
            a.abstained += 1
    return a


def gate(stats: dict[str, Any], *, min_precision: float = 0.90) -> dict[str, bool]:
    """Trust a class only when the lower 95% bound of its precision clears the bar (not just the point estimate)."""
    return {"success_trusted": (stats["success_precision_ci95"][0] >= min_precision) if stats["said_success"] else False,
            "failure_trusted": (stats["failure_precision_ci95"][0] >= min_precision) if stats["said_failure"] else False}
