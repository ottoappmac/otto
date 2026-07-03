"""End-of-run evaluation orchestrator for OTTO sessions.

Evaluates a completed session *once*: an LLM auto-selects appropriate
reference-free DeepEval metrics for the task, then each metric is scored
in-process via :mod:`tools.evaluation.evaluators`.  Results are written to
a per-session sidecar ``{session_id}.eval.json`` and summary fields are
stamped back onto the session meta so the UI can surface a score.

The orchestrator derives its input / output / tool-call data from the
durable on-disk transcript (``{session_id}.messages.json``), which is
written as the run streams and is unaffected by context compaction — so
no separate per-turn cache is needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Metrics that need only ``input`` + ``actual_output`` (no retrieval context,
# no ground-truth expected output).  Anything else in the registry needs data
# a completed OTTO run doesn't have, so the selector is constrained to these
# to guarantee every chosen metric can actually run.
_REFERENCE_FREE_METRICS = ("answer_relevancy", "toxicity", "bias", "g_eval")


# Error codes (as produced by ``backend.routes.sessions._friendly_error``) that
# indicate a transient / infrastructure failure a better prompt cannot fix.
# Anything else — including ``internal`` and unknown/``None`` — is treated as a
# candidate for LLM diagnosis.
_TRANSIENT_ERROR_CODES = frozenset({
    "mcp_connection", "llm_connection", "llm_rate_limit",
    "llm_auth", "llm_overloaded", "connection", "timeout",
})


def is_prompt_addressable(error_code: Optional[str]) -> bool:
    """Whether a failure with *error_code* might be avoided by a better prompt."""
    return (error_code or "internal") not in _TRANSIENT_ERROR_CODES


def classify_error_code(error_text: str) -> str:
    """Best-effort error-code derivation from a stored error string.

    Mirrors the substring checks in ``routes.sessions._friendly_error`` for
    contexts that only have the persisted error text (scheduler / trigger /
    crash-recovery), so the analyzer can classify failures consistently even
    when no authoritative code was captured at the point of failure.
    """
    low = (error_text or "").lower()
    if "closedresourceerror" in low:
        return "mcp_connection"
    if "timed out" in low or "timeout" in low or "apitimeouterror" in low:
        return "timeout"
    if "rate limit" in low or "rate_limit" in low or "429" in low:
        return "llm_rate_limit"
    if "authentication" in low or "could not resolve authentication" in low:
        return "llm_auth"
    if "overloaded" in low or "529" in low:
        return "llm_overloaded"
    if "apiconnectionerror" in low or "connection error" in low or "connection failed" in low:
        return "llm_connection"
    if "connecterror" in low:
        return "connection"
    return "internal"


# Generated-artifact inclusion: the agent's real deliverables often live in the
# session ``files/output`` directory (reports, charts, data) rather than the
# chat reply.  We fold their text content into ``actual_output`` so the judge
# scores what was actually produced, not just the chat summary.  Caps keep the
# judge prompt bounded.
_TEXT_ARTIFACT_EXTS = {
    ".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".log", ".html", ".htm", ".xml", ".py", ".js", ".css",
}
_PER_FILE_ARTIFACT_CHARS = 3000
_TOTAL_ARTIFACT_CHARS = 6000


def _strip_html(text: str) -> str:
    """Reduce an HTML document to readable text (titles, headings, labels).

    Drops ``<script>``/``<style>`` blocks and tags so a charts dashboard
    contributes its visible labels/headings to the judge rather than a wall of
    JavaScript that would otherwise dominate (and mislead) the score.
    """
    text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z]+;", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def _collect_output_artifacts(session_id: str) -> tuple[list[str], str]:
    """Gather text content of files in the session ``files/output`` directory.

    Returns ``(relative_names, formatted_context)``.  Binary files (images,
    PDFs, etc.) are listed by name/size only since the text judge can't read
    them.  Total content is capped so the judge prompt stays bounded.
    """
    try:
        from backend.session_manager import _session_files_dir
        base = _session_files_dir(session_id) / "output"
    except Exception:
        return [], ""
    if not base.exists() or not base.is_dir():
        return [], ""

    names: list[str] = []
    chunks: list[str] = []
    total = 0
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        names.append(rel)
        ext = p.suffix.lower()
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        if ext not in _TEXT_ARTIFACT_EXTS:
            chunks.append(f"### {rel} ({size} bytes, binary — content not shown)")
            continue
        if total >= _TOTAL_ARTIFACT_CHARS:
            chunks.append(f"### {rel} (omitted — artifact size budget reached)")
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            chunks.append(f"### {rel} (unreadable)")
            continue
        if ext in (".html", ".htm"):
            content = _strip_html(content)
        budget = min(_PER_FILE_ARTIFACT_CHARS, _TOTAL_ARTIFACT_CHARS - total)
        snippet = content[:budget]
        total += len(snippet)
        suffix = "\n…[truncated]" if len(content) > len(snippet) else ""
        chunks.append(f"### {rel}\n{snippet}{suffix}")

    if not names:
        return [], ""
    return names, "\n\n".join(chunks)


def _format_tool_trajectory(tools_called: list[dict[str, Any]], limit: int = 80) -> str:
    """Render the ordered tool-call sequence as a numbered list for the judge.

    Includes short argument *values* (truncated) — not just keys — so the judge
    can assess whether each call was appropriate (e.g. what was searched for,
    which file was written) rather than seeing opaque generic call names.
    """
    def _fmt_args(args: Any) -> str:
        if not isinstance(args, dict) or not args:
            return ""
        parts: list[str] = []
        for k, v in list(args.items())[:4]:
            sval = str(v).replace("\n", " ")
            if len(sval) > 80:
                sval = sval[:80] + "…"
            parts.append(f"{k}={sval}")
        return " (" + ", ".join(parts) + ")"

    lines: list[str] = []
    for i, tc in enumerate(tools_called[:limit], 1):
        name = tc.get("name") or "?"
        lines.append(f"{i}. {name}{_fmt_args(tc.get('input_parameters'))}")
    if len(tools_called) > limit:
        lines.append(f"… (+{len(tools_called) - limit} more calls)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidecar persistence
# ---------------------------------------------------------------------------

def _eval_path(session_id: str) -> Path:
    from backend.session_manager import _sessions_dir, _validate_session_id

    _validate_session_id(session_id)
    return _sessions_dir() / f"{session_id}.eval.json"


def load_evaluation(session_id: str) -> Optional[dict[str, Any]]:
    """Return the persisted evaluation sidecar, or ``None`` if absent."""
    try:
        p = _eval_path(session_id)
    except Exception:
        return None
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Corrupt eval sidecar: %s", p, exc_info=True)
        return None


def _write_evaluation(session_id: str, data: dict[str, Any]) -> None:
    p = _eval_path(session_id)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Transcript -> eval inputs
# ---------------------------------------------------------------------------

def _build_turns(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group a persisted transcript into (turns, tools_called).

    A *turn* is one user message and the agent's text reply to it::

        {"input": <user text>, "output": <agent text>, "tools": [<names>]}

    ``tools_called`` is the flat list of every tool call across the run
    (with arguments), used for trajectory-style metrics.
    """
    turns: list[dict[str, Any]] = []
    tools_called: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for msg in messages:
        mtype = msg.get("type")
        content = msg.get("content")
        if mtype == "user":
            if current is not None:
                turns.append(current)
            current = {"input": content if isinstance(content, str) else str(content),
                       "output": "", "tools": []}
        elif mtype == "agent":
            if current is not None and isinstance(content, str) and content.strip():
                current["output"] = content
        elif mtype == "tool_call":
            meta = msg.get("metadata") or {}
            name = msg.get("content") or meta.get("name") or ""
            tools_called.append({"name": name, "input_parameters": meta.get("args") or {}})
            if current is not None:
                current["tools"].append(name)

    if current is not None:
        turns.append(current)
    return turns, tools_called


# ---------------------------------------------------------------------------
# Judge model resolution (mirrors MemoryConfig.llm_family semantics)
# ---------------------------------------------------------------------------

def _resolve_judge_provider(family: str, main_provider: str) -> str:
    family = (family or "follow_main").lower()
    if family == "frontier":
        return "anthropic"
    if family == "mlx":
        return "mlx"
    return main_provider  # follow_main


_deepeval_env_configured = False


def _configure_deepeval_env() -> None:
    """Redirect DeepEval's on-disk state to a writable location.

    DeepEval stores its key/telemetry files in ``HIDDEN_DIR`` which defaults
    to ``.deepeval`` *relative to the current working directory* (resolved at
    ``deepeval`` import time via ``DEEPEVAL_CACHE_FOLDER``).  In the packaged
    app the CWD is the read-only bundle, so the first import would raise
    ``[Errno 30] Read-only file system: '.deepeval'`` and silently disable
    evaluation.  We point the cache/results dirs at the app-data dir and
    disable telemetry/dotenv before any deepeval import happens.

    Must run before the first ``import deepeval`` in the process to take
    effect (``HIDDEN_DIR`` is read once at module import).
    """
    global _deepeval_env_configured
    if _deepeval_env_configured:
        return
    try:
        from backend.config import get_app_data_dir

        cache_dir = get_app_data_dir() / "deepeval"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("DEEPEVAL_CACHE_FOLDER", str(cache_dir))
        os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", str(cache_dir))
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")
        os.environ.setdefault("DEEPEVAL_DISABLE_DOTENV", "1")
        os.environ.pop("CONFIDENT_API_KEY", None)
    except Exception:
        logger.debug("Could not configure DeepEval env", exc_info=True)
    finally:
        _deepeval_env_configured = True


def _build_judge() -> tuple[Any, Any]:
    """Build (raw_llm, deepeval_model) for the configured judge family.

    Returns ``(None, None)`` when no judge model can be constructed (e.g.
    Privacy Lock refuses a cloud provider) so the caller can mark the run
    ``skipped`` instead of silently producing nothing.
    """
    try:
        from backend.config import AppConfig
        from deep_agent.model_factory import create_llm
        from utilities.environment import Environment

        Environment.load()
        cfg = AppConfig.load()
        provider = _resolve_judge_provider(
            cfg.evaluation.llm_family, cfg.llm.provider or Environment.get_llm_provider()
        )
        raw_llm = create_llm(provider)
    except Exception as exc:
        logger.warning("Eval judge model unavailable: %s", exc)
        return None, None

    try:
        deepeval_model = _wrap_deepeval(raw_llm)
    except Exception as exc:
        logger.warning("Could not wrap judge model for DeepEval: %s", exc)
        return raw_llm, None

    return raw_llm, deepeval_model


def _coerce_to_schema(text: str, schema: Any) -> Any:
    """Best-effort parse of model *text* into a Pydantic *schema* instance.

    Returns the validated instance when possible, otherwise the original
    text so DeepEval can apply its own tolerant JSON parsing (and surface a
    meaningful error if that also fails).
    """
    try:
        data = _extract_json(text)
        if data is None:
            return text
        validate = getattr(schema, "model_validate", None)
        return validate(data) if callable(validate) else schema(**data)
    except Exception:
        return text


def _wrap_deepeval(raw_llm: Any) -> Any:
    """Wrap a ``BaseChatModel`` in a DeepEval ``DeepEvalBaseLLM`` adapter.

    DeepEval drives LLM-judged metrics (answer relevancy, g_eval, …) through
    ``(a_)generate_with_schema(prompt, schema=<pydantic model>)`` and only
    treats the response as trustworthy when it is an *instance* of that
    schema; otherwise it falls back to parsing the raw text as JSON via
    ``trimAndLoadJson`` and raises ``ValueError: Evaluation LLM outputted an
    invalid JSON`` the moment the model emits any prose, code fence, or
    truncated output.  We therefore honour the ``schema`` argument with
    LangChain structured output so the judge returns a validated object and
    never depends on the model emitting byte-perfect JSON.

    When a provider can't do structured output we fall back to plain text
    generation (normalised with ``extract_text_content`` so block-style
    Anthropic responses don't break things) plus a best-effort coercion into
    the requested schema.
    """
    _configure_deepeval_env()
    from deepeval.models import DeepEvalBaseLLM

    from backend.utils import extract_text_content

    class _Judge(DeepEvalBaseLLM):
        def __init__(self, llm: Any) -> None:
            self._llm = llm
            super().__init__("otto-eval-judge")

        def load_model(self) -> Any:
            return self._llm

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            if schema is not None:
                try:
                    return self._llm.with_structured_output(schema).invoke(prompt)
                except Exception as exc:
                    logger.debug(
                        "Structured judge output unavailable, using text: %s", exc
                    )
                    text = extract_text_content(self._llm.invoke(prompt).content)
                    return _coerce_to_schema(text, schema)
            return extract_text_content(self._llm.invoke(prompt).content)

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            if schema is not None:
                try:
                    return await self._llm.with_structured_output(schema).ainvoke(prompt)
                except Exception as exc:
                    logger.debug(
                        "Structured judge output unavailable, using text: %s", exc
                    )
                    resp = await self._llm.ainvoke(prompt)
                    return _coerce_to_schema(extract_text_content(resp.content), schema)
            resp = await self._llm.ainvoke(prompt)
            return extract_text_content(resp.content)

        def get_model_name(self, **kwargs: Any) -> str:
            return self.name

    return _Judge(raw_llm)


# ---------------------------------------------------------------------------
# Metric selection
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Any:
    """Best-effort extraction of the first JSON object/array from model text.

    Tolerates code fences and surrounding prose by slicing from the first
    opening bracket to the matching last closing bracket of the same kind.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    obj_start = text.find("{")
    arr_start = text.find("[")
    starts = [s for s in (obj_start, arr_start) if s != -1]
    if not starts:
        return None
    start = min(starts)
    close = "}" if text[start] == "{" else "]"
    end = text.rfind(close)
    if end <= start:
        return None
    # Drop trailing commas (``[1, 2,]`` / ``{"a": 1,}``) before parsing, matching
    # DeepEval's own tolerant loader so minor model formatting doesn't fail us.
    snippet = re.sub(r",\s*([\]}])", r"\1", text[start : end + 1])
    return json.loads(snippet)


def _candidate_metrics(has_tools: bool) -> list[str]:
    metrics = list(_REFERENCE_FREE_METRICS)
    if has_tools:
        # "trajectory" is a virtual type that expands to a g_eval at runtime
        # with pre-built criteria about tool selection and navigation efficiency.
        # It needs no ground-truth and is always meaningful when tools were used.
        metrics.append("trajectory")
    return metrics


async def _select_metrics(
    raw_llm: Any,
    *,
    task: str,
    output: str,
    tool_names: list[str],
    candidates: list[str],
    max_metrics: int,
    threshold: float,
    artifact_names: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Ask the judge which metrics fit this run.

    Returns ``(reasoning, selected)`` where *reasoning* is the model's
    plain-language explanation (shown live in the trace) and *selected* is
    the list of metric specs.  Falls back to answer relevancy if the model
    is unavailable or returns unparseable output.

    ``"trajectory"`` is a virtual candidate that is described in the catalog
    as a tool-use quality metric; the caller expands it before running.
    """
    from tools.evaluation.evaluators import REGISTRY

    fallback = [{"evaluator_type": "answer_relevancy", "threshold": threshold}]
    if raw_llm is None:
        return "No judge model — defaulting to Answer Relevancy.", fallback

    _TRAJECTORY_DESCRIPTION = (
        "Evaluates whether the agent selected and sequenced its tool calls "
        "appropriately to complete the task, and whether it avoided redundant "
        "or unnecessary calls."
    )

    def _catalog_entry(key: str) -> str:
        if key == "trajectory":
            return f"- trajectory: {_TRAJECTORY_DESCRIPTION}"
        info = REGISTRY.get(key)
        return f"- {key}: {info.description}" if info else ""

    catalog = "\n".join(e for k in candidates if (e := _catalog_entry(k)))

    tool_context = ""
    if tool_names:
        unique_tools = list(dict.fromkeys(tool_names))  # deduplicated, order preserved
        preview = ", ".join(unique_tools[:10])
        if len(unique_tools) > 10:
            preview += f" … ({len(unique_tools)} distinct tools total)"
        tool_context = (
            f"\nTool calls made ({len(tool_names)} total, "
            f"{len(unique_tools)} distinct): {preview}"
        )

    artifact_context = ""
    if artifact_names:
        a_preview = ", ".join(artifact_names[:8])
        if len(artifact_names) > 8:
            a_preview += f" (+{len(artifact_names) - 8} more)"
        artifact_context = (
            f"\nGenerated output files (deliverables saved to disk): {a_preview}"
        )

    prompt = (
        "You are an evaluation agent choosing which metrics to use to score "
        "an AI agent's response. Available metrics (pick 1 to "
        f"{max_metrics}):\n{catalog}\n\n"
        f"User request:\n{task[:1500]}\n\n"
        f"Agent response (excerpt):\n{output[:1500]}"
        f"{tool_context}{artifact_context}\n\n"
        "Respond with ONLY a JSON object with two keys:\n"
        '  "reasoning": a 1-3 sentence explanation of which metrics you are '
        "choosing and why, written in the first person as if thinking out loud.\n"
        '  "metrics": a JSON array where each element has "evaluator_type" '
        '(one of the listed ids) and, for "g_eval", also a "criteria" string.\n'
        'Example: {"reasoning":"The agent used 46 tool calls to automate a '
        'task, so I will check relevancy and run a trajectory evaluation.",'
        '"metrics":[{"evaluator_type":"answer_relevancy"},'
        '{"evaluator_type":"trajectory"}]}'
    )

    reasoning = ""
    try:
        from backend.utils import extract_text_content

        resp = await raw_llm.ainvoke(prompt)
        parsed_json = _extract_json(extract_text_content(resp.content))
        if isinstance(parsed_json, dict):
            reasoning = str(parsed_json.get("reasoning") or "").strip()
            metrics_raw = parsed_json.get("metrics")
        elif isinstance(parsed_json, list):
            # Some models ignore the wrapper and return a bare metrics array.
            metrics_raw = parsed_json
        else:
            metrics_raw = None
    except Exception as exc:
        logger.info("Eval metric selection fell back to default: %s", exc)
        return "Could not parse a metric plan — defaulting to Answer Relevancy.", fallback

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in metrics_raw if isinstance(metrics_raw, list) else []:
        if not isinstance(item, dict):
            continue
        etype = item.get("evaluator_type")
        if etype not in candidates or etype in seen:
            continue
        seen.add(etype)
        entry: dict[str, Any] = {"evaluator_type": etype, "threshold": threshold}
        if etype == "g_eval" and item.get("criteria"):
            entry["criteria"] = str(item["criteria"])
        selected.append(entry)
        if len(selected) >= max_metrics:
            break

    if not selected:
        return (reasoning or "No usable metric plan — defaulting to Answer Relevancy."), fallback
    return reasoning, selected


def _expand_trajectory(
    spec: dict[str, Any],
    tool_names: list[str],  # noqa: ARG001 — kept for signature stability
    task: str,
) -> dict[str, Any]:
    """Expand the virtual ``"trajectory"`` type into a concrete ``g_eval`` spec.

    The evaluated output (built by the orchestrator) contains the agent's final
    reply, a numbered ``TOOL CALL TRAJECTORY`` section listing the actual
    ordered calls, and any ``GENERATED OUTPUT FILES``. The criteria points the
    judge at those sections so it scores the real execution path and
    deliverables rather than complaining the trajectory is "missing".
    """
    criteria = (
        f"You are evaluating an AI agent's execution for this task: {task[:400]}\n\n"
        "The response contains an 'AGENT FINAL REPLY' section, a numbered "
        "'TOOL CALL TRAJECTORY' section (the actual ordered sequence of tool "
        "calls the agent made), and possibly a 'GENERATED OUTPUT FILES' section "
        "(its real deliverables such as report tables and charts saved to disk).\n\n"
        "Using the TOOL CALL TRAJECTORY, evaluate whether: (1) the tools chosen "
        "were appropriate for the task, (2) the sequence was logical and "
        "efficient, and (3) there were no obviously redundant or unnecessary "
        "calls. Also consider whether the GENERATED OUTPUT FILES satisfy what "
        "the task asked for. Score higher when the trajectory is tight and "
        "purposeful and the deliverables meet the request; lower when it is "
        "wasteful, confused, uses the wrong tools, or fails to produce the "
        "requested outputs."
    )
    return {"evaluator_type": "g_eval", "threshold": spec["threshold"], "criteria": criteria}


# ---------------------------------------------------------------------------
# Prompt-improvement suggestion (low-scoring runs)
# ---------------------------------------------------------------------------

async def _suggest_improved_prompt(
    raw_llm: Any,
    *,
    task: str,
    output: str,
    results: list[dict[str, Any]],
) -> Optional[tuple[str, str]]:
    """Ask the judge to propose a stronger user prompt for a low-scoring run.

    Returns ``(improved_prompt, reason)`` or ``None`` when the model is
    unavailable or returns unparseable output. Best-effort: never raises.
    """
    if raw_llm is None or not (task or "").strip():
        return None

    weaknesses: list[str] = []
    for r in results:
        reason = r.get("reason")
        if not reason:
            continue
        label = _METRIC_LABELS.get(r.get("evaluator_type"), r.get("evaluator_type"))
        score = r.get("score")
        pct = round(score * 100) if isinstance(score, (int, float)) else "?"
        weaknesses.append(f"- {label} ({pct}%): {reason}")
    weakness_text = "\n".join(weaknesses) or (
        "The response scored below the quality threshold."
    )

    prompt = (
        "You are an expert at writing prompts for AI agents. An agent's run "
        "scored below the quality threshold. Rewrite the user's ORIGINAL prompt "
        "so a future run is more likely to succeed: make the intent, scope, "
        "constraints and expected output explicit, while preserving the user's "
        "original goal. Do not answer or perform the task yourself.\n\n"
        f"ORIGINAL PROMPT:\n{task[:2000]}\n\n"
        f"WHY THE RUN SCORED LOW:\n{weakness_text[:2000]}\n\n"
        "Respond with ONLY a JSON object with two keys:\n"
        '  "improved_prompt": the full rewritten prompt the user could send.\n'
        '  "reason": a 1-2 sentence explanation of what you changed and why.'
    )

    try:
        from backend.utils import extract_text_content

        resp = await raw_llm.ainvoke(prompt)
        parsed = _extract_json(extract_text_content(resp.content))
    except Exception as exc:
        logger.info("Prompt-improvement suggestion failed: %s", exc)
        return None

    if not isinstance(parsed, dict):
        return None
    improved = str(parsed.get("improved_prompt") or "").strip()
    reason = str(parsed.get("reason") or "").strip()
    if not improved:
        return None
    return improved, reason


# ---------------------------------------------------------------------------
# Errored-run analysis (failed runs -> diagnosis + prompt fix)
# ---------------------------------------------------------------------------

async def _diagnose_and_suggest_prompt(
    raw_llm: Any,
    *,
    task: str,
    partial_output: str,
    error: str,
    tool_names: list[str],
) -> Optional[tuple[str, str, str]]:
    """Diagnose a failed run and, if prompt-addressable, propose a better prompt.

    Unlike :func:`_suggest_improved_prompt` (which acts on a low *score*), this
    works from the run's *error* plus whatever partial output / tool activity
    preceded it. The model first judges whether the failure is plausibly caused
    by the prompt; it only returns a rewrite when so.

    Returns ``(diagnosis, improved_prompt, reason)`` or ``None`` when the model
    is unavailable, the failure is not prompt-addressable, or output is
    unparseable. Best-effort: never raises.
    """
    if raw_llm is None or not (task or "").strip():
        return None

    tool_summary = ""
    if tool_names:
        uniq = list(dict.fromkeys(tool_names))
        preview = ", ".join(uniq[:12])
        if len(uniq) > 12:
            preview += f" (+{len(uniq) - 12} more)"
        tool_summary = f"\n\nTOOLS THE AGENT USED BEFORE FAILING: {preview}"

    partial = (partial_output or "").strip()
    partial_block = (
        f"\n\nAGENT'S PARTIAL OUTPUT BEFORE FAILING:\n{partial[:1500]}"
        if partial else ""
    )

    prompt = (
        "You are an expert at writing prompts for AI agents. An agent's run "
        "FAILED with an error. First decide whether the failure is plausibly "
        "caused by an unclear, under-specified, or overly broad USER PROMPT "
        "(e.g. ambiguous scope, missing inputs/paths/credentials the user "
        "should have provided, contradictory or impossible instructions). If "
        "so, rewrite the prompt so a future run is more likely to succeed, "
        "making intent, scope, constraints and expected output explicit while "
        "preserving the user's original goal. If the failure looks like an "
        "infrastructure, model, or tool problem unrelated to the prompt, say so "
        "and do NOT invent a rewrite. Do not perform the task yourself.\n\n"
        f"ORIGINAL PROMPT:\n{task[:2000]}\n\n"
        f"ERROR:\n{(error or 'Unknown error')[:1500]}"
        f"{tool_summary}{partial_block}\n\n"
        "Respond with ONLY a JSON object with these keys:\n"
        '  "diagnosis": 1-3 sentences on the likely cause of the failure.\n'
        '  "prompt_addressable": true if a better prompt could plausibly avoid '
        "this failure, false otherwise.\n"
        '  "improved_prompt": the full rewritten prompt (empty string when '
        "prompt_addressable is false).\n"
        '  "reason": 1-2 sentences on what you changed and why (empty when not '
        "addressable)."
    )

    try:
        from backend.utils import extract_text_content

        resp = await raw_llm.ainvoke(prompt)
        parsed = _extract_json(extract_text_content(resp.content))
    except Exception as exc:
        logger.info("Error-analysis suggestion failed: %s", exc)
        return None

    if not isinstance(parsed, dict):
        return None
    diagnosis = str(parsed.get("diagnosis") or "").strip()
    if not parsed.get("prompt_addressable"):
        # Surface the diagnosis to the caller via a sentinel so it can record
        # the "not prompt-fixable" explanation rather than a generic note.
        return (diagnosis, "", "") if diagnosis else None
    improved = str(parsed.get("improved_prompt") or "").strip()
    if not improved:
        return (diagnosis, "", "") if diagnosis else None
    reason = str(parsed.get("reason") or "").strip()
    return diagnosis, improved, reason


def _load_session_provenance(session_id: str) -> dict[str, Any]:
    """Read run-source fields from the session meta JSON (best-effort)."""
    from backend.session_manager import _sessions_dir

    meta_path = _sessions_dir() / f"{session_id}.json"
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {
        "trigger_source": data.get("trigger_source"),
        "schedule_id": data.get("schedule_id"),
        "trigger_id": data.get("trigger_id"),
        "agent_name": data.get("agent_name"),
        "title": data.get("title"),
    }


async def _create_eval_suggestion(
    session_id: str,
    *,
    improved_prompt: str,
    reason: str,
    overall: Optional[float],
) -> None:
    """Surface a low-score prompt improvement as a Suggestions-inbox hint.

    Manual runs become a re-runnable suggestion; scheduled / trigger runs become
    an "apply" suggestion that can update the stored schedule/trigger prompt.
    Best-effort: never raises.
    """
    prov = _load_session_provenance(session_id)
    trigger_source = prov.get("trigger_source")
    schedule_id = prov.get("schedule_id")
    trigger_id = prov.get("trigger_id")

    if trigger_source == "schedule" and schedule_id:
        target_kind, target_id = "schedule", schedule_id
    elif trigger_source == "trigger" and trigger_id:
        target_kind, target_id = "trigger", trigger_id
    else:
        target_kind, target_id = "manual", None

    run_label = prov.get("title") or session_id[:8]
    pct = round((overall or 0) * 100)
    if target_kind == "manual":
        title = f'Improve the prompt for "{run_label}"'
    else:
        title = f'Improve the {target_kind} prompt behind "{run_label}"'
    rationale = reason or (
        f"This run scored {pct}% — below the evaluation threshold. "
        "Here's a stronger prompt to try."
    )

    try:
        from backend.ambient_store import get_store

        store = await get_store()
        await store.add_eval_suggestion(
            {
                "title": title,
                "rationale": rationale,
                "proposed_prompt": improved_prompt,
                "suggested_agent": prov.get("agent_name"),
                "kind": "task",
                "sources": ["evaluation"],
                "session_id": session_id,
                "origin": "evaluation",
                "target_kind": target_kind,
                "target_id": target_id,
            }
        )
    except Exception:
        logger.debug("Failed to create eval suggestion", exc_info=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


_METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "toxicity": "Toxicity",
    "bias": "Bias",
    "g_eval": "G-Eval",
    "tool_correctness": "Tool Correctness",
    "trajectory": "Trajectory",
}


# Module-level task registry so fire-and-forget evals aren't garbage-collected
# before they finish (asyncio only holds a weak reference to running tasks).
_eval_tasks: set[Any] = set()


def launch_evaluation(session_id: str, *, manual: bool = False) -> Any:
    """Start an evaluation as a tracked background task and return it."""
    import asyncio

    task = asyncio.create_task(evaluate_session(session_id, manual=manual))
    _eval_tasks.add(task)
    task.add_done_callback(_eval_tasks.discard)
    return task


def launch_error_analysis(session_id: str, *, manual: bool = False) -> Any:
    """Start an errored-run analysis as a tracked background task and return it."""
    import asyncio

    task = asyncio.create_task(analyze_errored_session(session_id, manual=manual))
    _eval_tasks.add(task)
    task.add_done_callback(_eval_tasks.discard)
    return task


async def evaluate_session(session_id: str, *, manual: bool = False) -> dict[str, Any]:
    """Evaluate a completed session, emitting a live step-by-step trace.

    The evaluator's work (reading the conversation, choosing metrics and its
    reasoning, scoring each metric) is appended to a ``steps`` list that is
    persisted incrementally, so the UI can reveal it like a chat as it runs.

    Best-effort: any failure is captured into the sidecar (``status`` =
    ``error``/``skipped``) rather than raised, so callers (including the
    fire-and-forget completion hook) never crash.
    """
    from backend.config import AppConfig
    from backend.session_manager import _load_messages_async
    from tools.evaluation.evaluators import run_evaluation

    started = _now_iso()
    payload: dict[str, Any] = {
        "session_id": session_id, "status": "running", "manual": manual,
        "started_at": started, "steps": [], "results": [],
    }

    def _persist() -> dict[str, Any]:
        try:
            _write_evaluation(session_id, payload)
            _stamp_session_meta(session_id, payload)
        except Exception:
            logger.debug("Failed to persist evaluation", exc_info=True)
        return payload

    def _step(kind: str, text: str = "", **extra: Any) -> None:
        payload["steps"].append({"kind": kind, "text": text, "ts": _now_iso(), **extra})
        _persist()

    def _finish(status: str, **extra: Any) -> dict[str, Any]:
        payload.update(status=status, evaluated_at=_now_iso(), **extra)
        return _persist()

    _persist()  # initial running state

    try:
        messages = await _load_messages_async(session_id)
    except Exception as exc:
        _step("status", f"Could not load the transcript: {exc}")
        return _finish("error", error=f"Could not load transcript: {exc}")

    turns, tools_called = _build_turns(messages)
    user_input = turns[0]["input"] if turns else ""
    output = next((t["output"] for t in reversed(turns) if t["output"]), "")

    if not user_input or not output:
        _step("status", "This run has no input/output pair to evaluate.")
        return _finish("skipped", reason="Run has no input/output to evaluate.")

    payload["turns"] = turns
    _step(
        "status",
        f"Reading the conversation — {len(turns)} turn(s)"
        + (f", {len(tools_called)} tool call(s)" if tools_called else "")
        + ".",
    )

    # Fold generated output files into the evaluated output: the agent's real
    # deliverables (reports, chart dashboards, data) usually live on disk, not
    # in the chat reply, so without this the judge penalises "missing" tables /
    # charts that were in fact produced.
    artifact_names, artifact_text = await asyncio.to_thread(
        _collect_output_artifacts, session_id
    )
    output_for_eval = output
    if artifact_text:
        payload["artifacts"] = artifact_names
        preview = ", ".join(artifact_names[:5])
        if len(artifact_names) > 5:
            preview += f" (+{len(artifact_names) - 5} more)"
        _step("status", f"Including {len(artifact_names)} generated output file(s): {preview}.")
        output_for_eval = (
            f"{output}\n\n"
            "===== GENERATED OUTPUT FILES =====\n"
            "The agent saved its deliverables to these files (the report tables, "
            "charts and data live here — not only in the chat reply above):\n\n"
            f"{artifact_text}"
        )

    # Purpose-built output for the trajectory judge: the final reply, the actual
    # ordered tool-call sequence, and the generated deliverables, each clearly
    # sectioned so the g_eval criteria can reference them.
    traj_parts = [f"AGENT FINAL REPLY:\n{output}"]
    if tools_called:
        traj_parts.append(
            "TOOL CALL TRAJECTORY (actual ordered sequence):\n"
            + _format_tool_trajectory(tools_called)
        )
    if artifact_text:
        traj_parts.append("GENERATED OUTPUT FILES:\n" + artifact_text)
    trajectory_output = "\n\n".join(traj_parts)

    cfg = await AppConfig.aload()
    max_metrics = max(1, int(cfg.evaluation.max_metrics))
    threshold = float(cfg.evaluation.threshold)

    raw_llm, judge = _build_judge()
    if judge is None:
        if raw_llm is not None:
            # A judge LLM was built, but the DeepEval adapter couldn't be set
            # up (e.g. its on-disk cache dir wasn't writable). This is a setup
            # problem, not a missing provider — say so rather than blaming the
            # provider / Privacy Lock.
            msg = ("Judge model is available but DeepEval could not be "
                   "initialized (see backend logs).")
        else:
            msg = "No judge model is available (check provider / Privacy Lock)."
        _step("status", msg)
        return _finish("skipped", reason=msg)

    payload["model"] = getattr(judge, "name", None)
    _step("status", "Choosing which metrics to evaluate…")

    tool_names = [tc["name"] for tc in tools_called if tc.get("name")]
    candidates = _candidate_metrics(bool(tool_names))
    reasoning, selected = await _select_metrics(
        raw_llm, task=user_input, output=output_for_eval, tool_names=tool_names,
        artifact_names=artifact_names, candidates=candidates,
        max_metrics=max_metrics, threshold=threshold,
    )
    if reasoning:
        _step("thought", reasoning)

    # Expand virtual "trajectory" specs into concrete g_eval specs before display.
    expanded: list[tuple[str, dict[str, Any]]] = []
    for spec in selected:
        if spec["evaluator_type"] == "trajectory":
            expanded.append(("trajectory", _expand_trajectory(spec, tool_names, user_input)))
        else:
            expanded.append((spec["evaluator_type"], spec))

    names = ", ".join(_METRIC_LABELS.get(orig, orig) for orig, _ in expanded)
    payload["selected_metrics"] = [orig for orig, _ in expanded]
    _step("selection", f"Selected metrics: {names}.",
          metrics=[orig for orig, _ in expanded])

    results: list[dict[str, Any]] = []
    for orig_type, spec in expanded:
        etype = spec["evaluator_type"]  # actual DeepEval type (may be g_eval for trajectory)
        label = _METRIC_LABELS.get(orig_type, orig_type)
        # Pick the output view per metric:
        #  - trajectory: final reply + ordered tool calls + deliverables
        #  - other g_eval: chat reply + deliverables
        #  - statement-decomposition metrics (answer_relevancy, toxicity, bias):
        #    plain chat reply, so a large report/HTML blob doesn't overwhelm
        #    weaker local judges into invalid output.
        if orig_type == "trajectory":
            metric_output = trajectory_output
        elif etype == "g_eval":
            metric_output = output_for_eval
        else:
            metric_output = output
        _step("metric_start", f"Evaluating {label}…", metric=orig_type)
        try:
            result = await run_evaluation(
                etype,
                input=user_input,
                actual_output=metric_output,
                criteria=spec.get("criteria"),
                threshold=spec.get("threshold", threshold),
                model=judge,
            )
            result["evaluator_type"] = orig_type  # label as trajectory in results
            results.append(result)
            payload["results"] = results
            pct = round((result.get("score") or 0) * 100)
            verdict = "passed" if result.get("success") else "failed"
            _step("metric_result",
                  result.get("reason") or f"{label}: {pct}% ({verdict}).",
                  metric=orig_type, score=result.get("score"),
                  success=result.get("success"))
        except Exception as exc:
            logger.info("Metric %s failed: %s", orig_type, exc)
            err = {"evaluator_type": orig_type, "error": f"{type(exc).__name__}: {exc}"}
            results.append(err)
            payload["results"] = results
            _step("metric_result", f"{label} errored: {exc}", metric=orig_type)

    scored = [r["score"] for r in results if isinstance(r.get("score"), (int, float))]
    passed = sum(1 for r in results if r.get("success"))
    total = sum(1 for r in results if "success" in r)
    overall = round(sum(scored) / len(scored), 4) if scored else None

    # Below-threshold runs get an LLM-suggested improved prompt, shown on the
    # Evaluation tab and surfaced as a Suggestions-inbox hint.
    if overall is not None and overall < threshold:
        _step("status", "Score is below threshold — drafting a stronger prompt…")
        try:
            suggestion = await _suggest_improved_prompt(
                raw_llm, task=user_input, output=output_for_eval, results=results,
            )
        except Exception:
            logger.debug("Prompt suggestion errored", exc_info=True)
            suggestion = None
        if suggestion:
            improved_prompt, suggestion_reason = suggestion
            payload["suggested_prompt"] = improved_prompt
            payload["suggestion_reason"] = suggestion_reason
            _step(
                "suggestion",
                suggestion_reason or "Suggested an improved prompt for this run.",
                improved_prompt=improved_prompt,
            )
            await _create_eval_suggestion(
                session_id,
                improved_prompt=improved_prompt,
                reason=suggestion_reason,
                overall=overall,
            )

    _step(
        "done",
        f"Done — overall {round((overall or 0) * 100)}% · {passed}/{total} metric(s) passed."
        if overall is not None else "Done — no scoreable metrics.",
    )
    return _finish("done", overall_score=overall, pass_count=passed, total=total,
                   results=results)


async def analyze_errored_session(session_id: str, *, manual: bool = False) -> dict[str, Any]:
    """Analyze a run that ended in ``error`` and, when prompt-addressable,
    draft a stronger user prompt.

    Writes to the **same** ``{session_id}.eval.json`` sidecar as
    :func:`evaluate_session` (tagged ``kind="error_analysis"``) and emits the
    same incremental ``steps`` trace, so the existing Evaluation tab renders it
    unchanged. The flow:

    1. Read the persisted error + error_code from the session meta.
    2. Classify: transient/infra failures get a recorded "not prompt-fixable"
       note and stop early (no LLM call).
    3. Otherwise ask the judge LLM to diagnose the failure and, only if it
       attributes the failure to the prompt, rewrite it. A suggestion is
       surfaced on the Evaluation tab and in the Suggestions inbox.

    Best-effort: any failure is captured into the sidecar (``status`` =
    ``error``/``skipped``) rather than raised.
    """
    from backend.session_manager import _load_messages_async, _sessions_dir

    started = _now_iso()
    payload: dict[str, Any] = {
        "session_id": session_id, "status": "running", "manual": manual,
        "kind": "error_analysis", "started_at": started, "steps": [], "results": [],
    }

    def _persist() -> dict[str, Any]:
        try:
            _write_evaluation(session_id, payload)
            _stamp_session_meta(session_id, payload)
        except Exception:
            logger.debug("Failed to persist error analysis", exc_info=True)
        return payload

    def _step(kind: str, text: str = "", **extra: Any) -> None:
        payload["steps"].append({"kind": kind, "text": text, "ts": _now_iso(), **extra})
        _persist()

    def _finish(status: str, **extra: Any) -> dict[str, Any]:
        payload.update(status=status, evaluated_at=_now_iso(), **extra)
        return _persist()

    _persist()  # initial running state

    # Pull the failure details straight off the meta (error + classification).
    meta_path = _sessions_dir() / f"{session_id}.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        meta = {}
    error = str(meta.get("error") or "").strip()
    error_code = meta.get("error_code") or (classify_error_code(error) if error else None)
    payload["error"] = error
    payload["error_code"] = error_code

    _step("status", f"Analyzing a failed run — {error or 'unknown error'}.")

    try:
        messages = await _load_messages_async(session_id)
    except Exception as exc:
        _step("status", f"Could not load the transcript: {exc}")
        return _finish("error", error=f"Could not load transcript: {exc}")

    turns, tools_called = _build_turns(messages)
    user_input = turns[0]["input"] if turns else ""
    partial_output = next((t["output"] for t in reversed(turns) if t["output"]), "")
    tool_names = [tc["name"] for tc in tools_called if tc.get("name")]

    if not user_input:
        _step("status", "This run has no user prompt to improve.")
        return _finish("skipped", reason="Run has no user prompt to analyze.")

    payload["turns"] = turns

    # Transient / infrastructure failures can't be fixed by a better prompt —
    # record that plainly and skip the (pointless) LLM call.
    if not is_prompt_addressable(error_code):
        msg = (
            f"This run failed with a {error_code or 'transient'} error "
            "(infrastructure / model / connectivity), which a better prompt "
            "cannot fix — no suggestion generated."
        )
        _step("status", msg)
        return _finish("skipped", reason=msg)

    raw_llm, _judge = _build_judge()
    if raw_llm is None:
        msg = "No judge model is available (check provider / Privacy Lock)."
        _step("status", msg)
        return _finish("skipped", reason=msg)

    _step("status", "Diagnosing the failure and drafting a stronger prompt…")
    try:
        suggestion = await _diagnose_and_suggest_prompt(
            raw_llm,
            task=user_input,
            partial_output=partial_output,
            error=error,
            tool_names=tool_names,
        )
    except Exception:
        logger.debug("Error-analysis suggestion errored", exc_info=True)
        suggestion = None

    if suggestion is None:
        msg = "Could not attribute this failure to the prompt — no suggestion generated."
        _step("status", msg)
        return _finish("done", reason=msg)

    diagnosis, improved_prompt, reason = suggestion
    if diagnosis:
        payload["diagnosis"] = diagnosis
        _step("thought", diagnosis)

    # The model judged the failure not prompt-addressable (rewrite is empty).
    if not improved_prompt:
        msg = "This failure is not attributable to the prompt — no suggestion generated."
        _step("status", msg)
        return _finish("done", reason=diagnosis or msg)

    payload["suggested_prompt"] = improved_prompt
    payload["suggestion_reason"] = reason
    _step(
        "suggestion",
        reason or "Suggested an improved prompt for this failed run.",
        improved_prompt=improved_prompt,
    )

    await _create_eval_suggestion(
        session_id,
        improved_prompt=improved_prompt,
        reason=reason or (
            f"This run failed ({error or 'unknown error'}). "
            "Here's a stronger prompt to try."
        ),
        overall=None,
    )

    _step("done", "Done — drafted an improved prompt based on the failure.")
    return _finish("done")


def _stamp_session_meta(session_id: str, payload: dict[str, Any]) -> None:
    """Write eval summary fields into the session meta JSON so /api/runs
    surfaces them without re-reading the eval sidecar.

    Also updates the in-memory ``Session`` (when still loaded) so a later
    ``save_meta()`` on a subsequent turn doesn't wipe the fields.
    """
    from backend.session_manager import _sessions_dir

    status = payload.get("status")
    overall = payload.get("overall_score")
    passes = payload.get("pass_count")
    total = payload.get("total")

    meta_path = _sessions_dir() / f"{session_id}.json"
    if meta_path.exists():
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            data["eval_status"] = status
            data["eval_overall_score"] = overall
            data["eval_pass_count"] = passes
            data["eval_total"] = total
            meta_path.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception:
            logger.debug("Failed to stamp eval fields onto session meta", exc_info=True)

    try:
        from backend.state import session_mgr

        live = session_mgr.get_session(session_id)
        if live is not None:
            live.eval_status = status
            live.eval_overall_score = overall
            live.eval_pass_count = passes
            live.eval_total = total
    except Exception:
        logger.debug("Failed to update in-memory session eval fields", exc_info=True)
