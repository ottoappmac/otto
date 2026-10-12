"""HTTP client for a Jev-compatible ``POST /v1/systemone`` server.

One question per call. OpenJev and TypeSafe Jev share this wire format.
OpenJev's ``confidence`` is ``1 - H(p)/ln K`` (output entropy), not a
calibrated probability. ``noul`` answers often omit confidence; that
value is derived as ``abs(2p - 1)``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

STATE_CHAR_LIMIT = 12_000
MAX_OPTIONS = 12
REQUEST_TIMEOUT_S = 20.0
_TRUNCATION_MARKER = "\n\n[truncated]"
_KEY_RE = re.compile(r"[^a-z0-9_]+")


class DecisionClientError(Exception):
    """The Jev server answered, but the body could not be used."""


@dataclass
class DecisionAnswer:
    kind: str
    answer: str
    confidence: float
    margin: float
    probabilities: dict[str, float] = field(default_factory=dict)


def truncate_state(state: str, limit: int = STATE_CHAR_LIMIT) -> str:
    text = state or ""
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _TRUNCATION_MARKER


def sanitize_key(raw: str, used: set[str]) -> str:
    """Turn a free-text option into a stable choice key."""
    key = _KEY_RE.sub("_", (raw or "").strip().lower()).strip("_")
    key = re.sub(r"_+", "_", key)[:48] or "option"
    base = key
    n = 2
    while key in used:
        suffix = f"_{n}"
        key = base[: 48 - len(suffix)] + suffix
        n += 1
    used.add(key)
    return key


def option_map(options: Any) -> dict[str, str]:
    """Coerce a list, a JSON-ish string, or a dict into ``{key: description}``."""
    if options is None:
        return {}
    if isinstance(options, dict):
        used: set[str] = set()
        out: dict[str, str] = {}
        for raw_key, raw_desc in options.items():
            key = sanitize_key(str(raw_key), used)
            desc = str(raw_desc).strip() or key
            out[key] = desc
        return out
    items = _coerce_list(options)
    used = set()
    out = {}
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        key = sanitize_key(text, used)
        out[key] = text
    return out


def _coerce_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return [value]
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        import json
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            return parsed
    for sep in ("\n", ";", "|"):
        if sep in text:
            parts = [p.strip() for p in text.split(sep) if p.strip()]
            if len(parts) > 1:
                return parts
    if "," in text:
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) > 1:
            return parts
    return [text]


def source_label(base_url: str) -> str:
    """Short name for the result text: local, typesafe, or another remote host."""
    host = (urlparse(base_url).hostname or base_url).lower()
    if host in {"127.0.0.1", "localhost", "::1"}:
        kind = "local"
    elif host.endswith("typesafe.ai"):
        kind = "typesafe"
    else:
        kind = "remote"
    return f"{kind} ({host})"


def model_name_for(base_url: str) -> str:
    """Wire model id. Never send a pinned Jev version to OpenJev (it 400s)."""
    host = (urlparse(base_url).hostname or "").lower()
    if host.endswith("typesafe.ai"):
        return "jev-latest"
    return "openjev-latest"


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def entropy_confidence(probabilities: dict[str, float]) -> float:
    vals = [float(p) for p in probabilities.values() if float(p) > 0]
    k = len(probabilities)
    if k <= 1 or not vals:
        return 1.0
    entropy = -sum(p * math.log(p) for p in vals)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(k)))


def _margin(probabilities: dict[str, float]) -> float:
    ordered = sorted((float(p) for p in probabilities.values()), reverse=True)
    if not ordered:
        return 0.0
    second = ordered[1] if len(ordered) > 1 else 0.0
    return max(0.0, ordered[0] - second)


def verdict_text(
    confidence: float,
    *,
    confident_threshold: float,
    unsure_threshold: float,
    unattended: bool,
) -> str:
    if confidence >= confident_threshold:
        return "CONFIDENT. Proceed. Advisory only; approvals still apply."
    if confidence >= unsure_threshold:
        return "MODERATE. Do a cheap check first. Advisory only; approvals still apply."
    if unattended:
        return (
            "UNSURE. No user is available; take the safest reversible option "
            "or stop and report."
        )
    return "UNSURE. Call `ask_user`."


def parse_answer(payload: dict[str, Any]) -> DecisionAnswer:
    answers = payload.get("answers") or {}
    if not isinstance(answers, dict) or not answers:
        raise DecisionClientError("Jev server returned no answers")
    body = next(iter(answers.values()))
    if not isinstance(body, dict):
        raise DecisionClientError("Jev server returned a malformed answer")
    if "noul" in body:
        p_yes = float(body["noul"])
        p_yes = max(0.0, min(1.0, p_yes))
        probabilities = {"yes": p_yes, "no": 1.0 - p_yes}
        confidence = body.get("confidence")
        if confidence is None:
            confidence = abs(2.0 * p_yes - 1.0)
        answer = "yes" if p_yes >= 0.5 else "no"
        return DecisionAnswer(
            kind="noul",
            answer=answer,
            confidence=float(confidence),
            margin=_margin(probabilities),
            probabilities=probabilities,
        )
    if "choice" in body:
        probabilities = {
            str(k): float(v) for k, v in (body.get("probabilities") or {}).items()
        }
        confidence = body.get("confidence")
        if confidence is None:
            confidence = entropy_confidence(probabilities)
        return DecisionAnswer(
            kind="choice",
            answer=str(body["choice"]),
            confidence=float(confidence),
            margin=_margin(probabilities),
            probabilities=probabilities,
        )
    if "score" in body:
        probabilities = {
            str(k): float(v) for k, v in (body.get("probabilities") or {}).items()
        }
        confidence = body.get("confidence")
        if confidence is None:
            confidence = entropy_confidence(probabilities)
        legend = body.get("legend") or {}
        score = float(body["score"])
        label = _nearest_legend(score, legend)
        shown = f"{score:.2f}" if label is None else f"{score:.2f} ({label})"
        return DecisionAnswer(
            kind="score",
            answer=shown,
            confidence=float(confidence),
            margin=_margin(probabilities),
            probabilities=probabilities,
        )
    raise DecisionClientError("Jev server returned an unknown answer type")


def _nearest_legend(score: float, legend: Any) -> str | None:
    if not isinstance(legend, dict) or not legend:
        return None
    best_key: str | None = None
    best_dist = float("inf")
    for key in legend:
        try:
            dist = abs(float(key) - score)
        except (TypeError, ValueError):
            continue
        if dist < best_dist:
            best_dist = dist
            best_key = str(key)
    if best_key is None:
        return None
    return str(legend[best_key])


def build_question(
    question: str,
    options: Any = None,
    scale: Any = None,
) -> dict[str, Any]:
    """One typed question. ``scale`` wins over ``options``; neither means yes/no."""
    levels = _coerce_list(scale) if scale else []
    levels = [str(level).strip() for level in levels if str(level).strip()]
    if len(levels) >= 2:
        if len(levels) > 10:
            raise DecisionClientError("a score scale can have at most 10 levels")
        return {
            "type": "score",
            "instructions": question,
            "criteria": levels,
        }
    mapping = option_map(options)
    if len(mapping) > MAX_OPTIONS:
        raise DecisionClientError(
            f"at most {MAX_OPTIONS} options per decision; narrow the list"
        )
    if mapping:
        return {
            "type": "choice",
            "instructions": question,
            "criteria": mapping,
        }
    return {"type": "noul", "instructions": question}


def ask(
    state: str,
    question: str,
    options: Any = None,
    scale: Any = None,
    *,
    base_url: str,
    api_key: str,
    timeout: float = REQUEST_TIMEOUT_S,
    http_client: httpx.Client | None = None,
) -> DecisionAnswer:
    """POST one question. Raises :class:`DecisionClientError` or ``httpx.HTTPError``."""
    q = build_question(question, options, scale)
    body = {
        "model": model_name_for(base_url),
        "state": truncate_state(state),
        "questions": {"q": q},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/v1/systemone"
    owns = http_client is None
    client = http_client or httpx.Client(timeout=timeout)
    try:
        response = client.post(url, json=body, headers=headers, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns:
            client.close()
    if not isinstance(payload, dict):
        raise DecisionClientError("Jev server returned a non-object body")
    return parse_answer(payload)


def format_answer(
    result: DecisionAnswer,
    *,
    confident_threshold: float,
    unsure_threshold: float,
    unattended: bool,
    source: str = "",
) -> str:
    probs = " | ".join(
        f"{name} {prob:.2f}"
        for name, prob in sorted(result.probabilities.items(), key=lambda kv: -kv[1])
    )
    verdict = verdict_text(
        result.confidence,
        confident_threshold=confident_threshold,
        unsure_threshold=unsure_threshold,
        unattended=unattended,
    )
    lines = [
        f"jev: {result.answer} (confidence {result.confidence:.2f}, margin {result.margin:.2f})",
    ]
    if probs:
        lines.append(f"  {probs}")
    lines.append(f"verdict: {verdict}")
    if source:
        lines.append(f"source: {source}")
    return "\n".join(lines)
