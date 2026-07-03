"""Evaluator registry and runner.

Maps evaluator type identifiers to DeepEval metric instances and provides
a unified interface for running evaluations.  All DeepEval imports are
lazy so this module can be loaded without deepeval installed (e.g. for
registry introspection).

Model resolution reuses the app's own ``create_llm()`` via a thin adapter
(``EvalModel``) so authentication — Anthropic direct, Bedrock with
access keys — is handled identically to the main agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


def _extract_text(content: Any) -> str:
    """Return the plain-text portion of an LLM message content block.

    Handles both the simple string form and the list-of-dicts form used by
    models that interleave text with tool-use blocks (e.g. Anthropic).
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def _extract_json(text: str) -> Any:
    """Best-effort extraction of the first JSON object/array from model text."""
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
    snippet = re.sub(r",\s*([\]}])", r"\1", text[start : end + 1])
    return json.loads(snippet)


def _coerce_to_schema(text: str, schema: Any) -> Any:
    """Best-effort parse of model *text* into a Pydantic *schema* instance.

    Returns the validated instance when possible, otherwise the original text
    so DeepEval can apply its own tolerant JSON parsing.
    """
    try:
        data = _extract_json(text)
        if data is None:
            return text
        validate = getattr(schema, "model_validate", None)
        return validate(data) if callable(validate) else schema(**data)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Registry — lightweight metadata, no external imports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluatorInfo:
    """Metadata for a registered evaluator type."""

    name: str
    description: str
    requires_context: bool = False
    requires_expected_output: bool = False
    requires_tools: bool = False
    requires_criteria: bool = False
    is_llm_based: bool = True


REGISTRY: dict[str, EvaluatorInfo] = {
    "answer_relevancy": EvaluatorInfo(
        name="Answer Relevancy",
        description="Measures how relevant the response is to the input query.",
    ),
    "faithfulness": EvaluatorInfo(
        name="Faithfulness",
        description="Measures whether the response is grounded in the provided context.",
        requires_context=True,
    ),
    "hallucination": EvaluatorInfo(
        name="Hallucination",
        description="Detects hallucinated or unsupported claims in the response.",
        requires_context=True,
    ),
    "contextual_precision": EvaluatorInfo(
        name="Contextual Precision",
        description=(
            "Measures whether relevant context nodes are ranked higher "
            "than irrelevant ones."
        ),
        requires_context=True,
        requires_expected_output=True,
    ),
    "contextual_recall": EvaluatorInfo(
        name="Contextual Recall",
        description=(
            "Measures whether all ground-truth claims can be attributed "
            "to the retrieved context."
        ),
        requires_context=True,
        requires_expected_output=True,
    ),
    "contextual_relevancy": EvaluatorInfo(
        name="Contextual Relevancy",
        description="Measures relevancy of the retrieved context to the input query.",
        requires_context=True,
    ),
    "toxicity": EvaluatorInfo(
        name="Toxicity",
        description="Detects toxic, offensive, or harmful content in the response.",
    ),
    "bias": EvaluatorInfo(
        name="Bias",
        description="Detects biased opinions or language in the response.",
    ),
    "g_eval": EvaluatorInfo(
        name="G-Eval",
        description=(
            "Configurable LLM-as-judge with custom natural-language criteria "
            "and probability-weighted scoring."
        ),
        requires_criteria=True,
    ),
    "summarization": EvaluatorInfo(
        name="Summarization",
        description="Evaluates summary quality against the source text.",
    ),
    "tool_correctness": EvaluatorInfo(
        name="Tool Correctness",
        description=(
            "Evaluates whether the agent selected the correct tools "
            "with correct parameters."
        ),
        requires_tools=True,
    ),
}


# ---------------------------------------------------------------------------
# Model adapter — wraps create_llm() for DeepEval
# ---------------------------------------------------------------------------

def _build_eval_model_class() -> type:
    """Build a ``DeepEvalBaseLLM`` subclass that wraps ``create_llm()``.

    Defined as a factory so the ``deepeval`` import is deferred until the
    class is actually needed, keeping the module importable without deepeval.
    """
    from deepeval.models import DeepEvalBaseLLM

    class EvalModel(DeepEvalBaseLLM):
        """Adapter exposing a ``BaseChatModel`` as a DeepEval model.

        Single source of truth — delegates to
        ``deep_agent.model_factory.create_llm()`` which already handles
        Anthropic direct and Bedrock with access keys.

        DeepEval scores LLM-judged metrics through
        ``(a_)generate_with_schema(prompt, schema=<pydantic model>)`` and only
        trusts a response that is an *instance* of that schema; otherwise it
        parses the raw text as JSON and raises ``ValueError: Evaluation LLM
        outputted an invalid JSON`` on any prose/fence/truncation.  We honour
        ``schema`` via LangChain structured output (with a text + coercion
        fallback) so judging never hinges on byte-perfect JSON.
        """

        def __init__(self) -> None:
            from utilities.environment import Environment

            Environment.load()
            provider = Environment.get_llm_provider()

            from deep_agent.model_factory import create_llm

            self._llm = create_llm(provider)
            model_name = Environment.get_anthropic_model_name()

            super().__init__(model_name)

        def load_model(self) -> Any:
            return self._llm

        def generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            if schema is not None:
                try:
                    return self._llm.with_structured_output(schema).invoke(prompt)
                except Exception as exc:
                    logger.debug(
                        "Structured eval output unavailable, using text: %s", exc
                    )
                    text = _extract_text(self._llm.invoke(prompt).content)
                    return _coerce_to_schema(text, schema)
            return _extract_text(self._llm.invoke(prompt).content)

        async def a_generate(self, prompt: str, schema: Any = None, **kwargs: Any) -> Any:
            if schema is not None:
                try:
                    return await self._llm.with_structured_output(schema).ainvoke(prompt)
                except Exception as exc:
                    logger.debug(
                        "Structured eval output unavailable, using text: %s", exc
                    )
                    resp = await self._llm.ainvoke(prompt)
                    return _coerce_to_schema(_extract_text(resp.content), schema)
            resp = await self._llm.ainvoke(prompt)
            return _extract_text(resp.content)

        def get_model_name(self, **kwargs: Any) -> str:
            return self.name

    return EvalModel


def resolve_model() -> Any | None:
    """Create an evaluator model using the app's own LLM factory.

    Supports every provider mode the main agent supports
    (Anthropic direct, Bedrock access-keys) because it
    delegates to ``create_llm()`` under the hood.

    Returns ``None`` when model creation fails — only deterministic
    metrics will work in that case.
    """
    try:
        cls = _build_eval_model_class()
        return cls()
    except Exception as exc:
        logger.warning("Could not create evaluator model: %s", exc)
        logger.info("Only deterministic evaluators (tool_correctness) will work")
        return None


# ---------------------------------------------------------------------------
# Metric factory
# ---------------------------------------------------------------------------

def _build_metric_kwargs(
    model: Any | None,
    threshold: float,
    is_llm_based: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"threshold": threshold}
    if model is not None and is_llm_based:
        kwargs["model"] = model
    return kwargs


def create_metric(
    evaluator_type: str,
    *,
    model: Any | None = None,
    threshold: float = 0.5,
    criteria: str | None = None,
) -> Any:
    """Create a DeepEval metric instance for *evaluator_type*."""
    info = REGISTRY.get(evaluator_type)
    if info is None:
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type!r}. "
            f"Available: {sorted(REGISTRY)}"
        )

    if info.is_llm_based and model is None:
        raise RuntimeError(
            f"Evaluator {evaluator_type!r} requires an LLM but no model "
            "could be created.  Check LLM provider settings in the app."
        )

    kwargs = _build_metric_kwargs(model, threshold, info.is_llm_based)

    # Lazy-import each metric class only when needed.
    if evaluator_type == "answer_relevancy":
        from deepeval.metrics import AnswerRelevancyMetric
        return AnswerRelevancyMetric(**kwargs)

    if evaluator_type == "faithfulness":
        from deepeval.metrics import FaithfulnessMetric
        return FaithfulnessMetric(**kwargs)

    if evaluator_type == "hallucination":
        from deepeval.metrics import HallucinationMetric
        return HallucinationMetric(**kwargs)

    if evaluator_type == "contextual_precision":
        from deepeval.metrics import ContextualPrecisionMetric
        return ContextualPrecisionMetric(**kwargs)

    if evaluator_type == "contextual_recall":
        from deepeval.metrics import ContextualRecallMetric
        return ContextualRecallMetric(**kwargs)

    if evaluator_type == "contextual_relevancy":
        from deepeval.metrics import ContextualRelevancyMetric
        return ContextualRelevancyMetric(**kwargs)

    if evaluator_type == "toxicity":
        from deepeval.metrics import ToxicityMetric
        return ToxicityMetric(**kwargs)

    if evaluator_type == "bias":
        from deepeval.metrics import BiasMetric
        return BiasMetric(**kwargs)

    if evaluator_type == "g_eval":
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCaseParams
        return GEval(
            name="custom_criteria",
            criteria=criteria or "Overall quality and helpfulness of the response",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            **kwargs,
        )

    if evaluator_type == "summarization":
        from deepeval.metrics import SummarizationMetric
        return SummarizationMetric(**kwargs)

    if evaluator_type == "tool_correctness":
        from deepeval.metrics import ToolCorrectnessMetric
        return ToolCorrectnessMetric(**kwargs)

    raise ValueError(f"No factory registered for evaluator type: {evaluator_type!r}")


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def _build_test_case(
    *,
    input: str,
    actual_output: str,
    expected_output: str | None = None,
    context: list[str] | None = None,
    retrieval_context: list[str] | None = None,
    tools_called: list[dict[str, Any]] | None = None,
    expected_tools: list[dict[str, Any]] | None = None,
) -> Any:
    """Build a ``deepeval.test_case.LLMTestCase`` from raw parameters."""
    from deepeval.test_case import LLMTestCase

    tc_kwargs: dict[str, Any] = {
        "input": input,
        "actual_output": actual_output,
    }
    if expected_output:
        tc_kwargs["expected_output"] = expected_output
    if context:
        tc_kwargs["context"] = context
    if retrieval_context:
        tc_kwargs["retrieval_context"] = retrieval_context

    if tools_called:
        from deepeval.test_case import ToolCall
        tc_kwargs["tools_called"] = [
            ToolCall(
                name=t["name"],
                input_parameters=t.get("input_parameters", {}),
            )
            for t in tools_called
        ]
    if expected_tools:
        from deepeval.test_case import ToolCall
        tc_kwargs["expected_tools"] = [
            ToolCall(
                name=t["name"],
                input_parameters=t.get("input_parameters", {}),
            )
            for t in expected_tools
        ]

    return LLMTestCase(**tc_kwargs)


async def run_evaluation(
    evaluator_type: str,
    *,
    input: str,
    actual_output: str,
    expected_output: str | None = None,
    context: list[str] | None = None,
    retrieval_context: list[str] | None = None,
    tools_called: list[dict[str, Any]] | None = None,
    expected_tools: list[dict[str, Any]] | None = None,
    criteria: str | None = None,
    threshold: float = 0.5,
    model: Any | None = None,
) -> dict[str, Any]:
    """Run a single evaluation and return structured results.

    Returns a dict with ``score``, ``reason``, ``success``,
    ``evaluator_type``, and ``threshold``.
    """
    test_case = _build_test_case(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        context=context,
        retrieval_context=retrieval_context,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )

    metric = create_metric(
        evaluator_type,
        model=model,
        threshold=threshold,
        criteria=criteria,
    )

    try:
        await metric.a_measure(test_case)
    except (NotImplementedError, AttributeError):
        await asyncio.to_thread(metric.measure, test_case)

    return {
        "score": metric.score,
        "reason": getattr(metric, "reason", None),
        "success": metric.is_successful(),
        "evaluator_type": evaluator_type,
        "threshold": threshold,
    }
