"""Labels and admission: which label counts, and what an admitted run contributes.

The research is consistent that admission, not retrieval, decides whether experience memory helps: prompted judges
reach about 0.70 precision on "success", agents claim success on most failed runs, and memories admitted on
self-judgement were 50-60% failures in a 2026 budget-matched study (see the report linked from the plan). So:

* Sources, most trusted first: ``user`` > ``benchmark`` > ``verified`` > ``judge`` > ``self_report``.
* ``self_report`` never admits anything. ``judge`` admits only when calibration says it may
  (``ProceduralMemoryConfig.judge_success_trusted`` / ``judge_failure_trusted``, set after Phase 0 shows >= 0.90
  precision on our own runs).
* A **successful** run contributes its whole-run recipe, its segments, facts, lessons and scripts.
* A **failed** run never contributes its step sequence as a whole. It contributes the segments the trace itself
  confirmed (not the last one, where it went wrong), facts, and lessons phrased as checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SOURCE_RANK = {"user": 5, "benchmark": 4, "verified": 3, "judge": 2, "self_report": 1}
LABELS = ("success", "failure", "unknown")


@dataclass(frozen=True)
class Policy:
    judge_success_trusted: bool = False
    judge_failure_trusted: bool = False
    segments_from_failed_runs: bool = True
    min_actions: int = 2


@dataclass
class Decision:
    status: str                 # admitted | pending | excluded
    label: str
    source: str | None
    confidence: float | None
    reason: str


def effective_label(labels: list[dict[str, Any]]) -> tuple[str, str | None, float | None]:
    """(label, source, confidence) from the label history: the most trusted source wins, then the latest."""
    best = None
    for lab in labels:
        value = lab.get("value")
        if value not in LABELS or value == "unknown":
            continue
        key = (SOURCE_RANK.get(lab.get("source") or "", 0), lab.get("at") or 0)
        if best is None or key > best[0]:
            best = (key, value, lab.get("source"), lab.get("confidence"))
    if best is None:
        return "unknown", None, None
    return best[1], best[2], best[3]


def decide(labels: list[dict[str, Any]], *, actions: int, infra_error: bool, policy: Policy) -> Decision:
    if infra_error:
        return Decision("excluded", "unknown", None, None, "infrastructure error: not the agent's outcome")
    if actions < policy.min_actions:
        return Decision("excluded", "unknown", None, None, f"fewer than {policy.min_actions} actions")
    label, source, conf = effective_label(labels)
    if label == "unknown" or source is None:
        return Decision("pending", "unknown", None, None, "no label yet")
    if source == "self_report":
        return Decision("pending", label, source, conf, "the agent's own claim admits nothing")
    if source == "judge":
        if label == "success" and not policy.judge_success_trusted:
            return Decision("pending", label, source, conf, "judge not calibrated for successes")
        if label == "failure" and not policy.judge_failure_trusted:
            return Decision("pending", label, source, conf, "judge not calibrated for failures")
    return Decision("admitted", label, source, conf, f"{label} from {source}")


def admitted_segments(segments: list[dict[str, Any]], run_label: str, policy: Policy) -> list[dict[str, Any]]:
    """The segments a run may contribute as recipes, each with its label and where that label came from."""
    out = []
    if run_label == "success":
        for s in segments:
            if s.get("recipe"):
                out.append({**s, "seg_label": "success", "seg_source": "run"})
        return out
    if run_label != "failure" or not policy.segments_from_failed_runs:
        return out
    for k, s in enumerate(segments):
        last = k == len(segments) - 1
        if last or s.get("label") != "success" or not s.get("recipe"):
            continue
        out.append({**s, "seg_label": "success", "seg_source": "segment_evidence"})
    return out
