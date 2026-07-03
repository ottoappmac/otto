#!/usr/bin/env python3
"""Agent Evaluator Service MCP Server.

Standalone MCP server that exposes evaluation metrics as tools for
scoring agent sessions.  Designed to run as a managed child process
alongside the backend, but can also be used independently by
any MCP client.

Usage::

    # Streamable HTTP (default)
    python -m tools.evaluation.mcp_server --port 8941

    # stdio (for direct MCP client piping)
    python -m tools.evaluation.mcp_server --transport stdio
"""

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure src/ is importable when run directly as a script.
_src_dir = str(Path(__file__).resolve().parent.parent.parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Hard-disable all DeepEval telemetry and cloud reporting.
# DEEPEVAL_TELEMETRY_OPT_OUT  — disables anonymous usage analytics.
# DEEPEVAL_DISABLE_DOTENV     — prevents loading .env files that might
#                                contain a CONFIDENT_API_KEY (which would
#                                auto-upload results to Confident AI).
# CONFIDENT_API_KEY            — explicitly blanked so no cloud uploads
#                                can occur even if inherited from parent env.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "1"
os.environ["DEEPEVAL_DISABLE_DOTENV"] = "1"
os.environ.pop("CONFIDENT_API_KEY", None)

from mcp.server.fastmcp import FastMCP  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("agent-eval-service")

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8941

mcp = FastMCP(
    "agent-eval-service",
    host=_DEFAULT_HOST,
    port=_DEFAULT_PORT,
    instructions=(
        "Agent evaluation server.  Use `list_evaluators` to discover "
        "available metrics, then call `evaluate` or `evaluate_trajectory` "
        "to score agent outputs."
    ),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_model_cache: dict[str, Any] = {}


def _get_model() -> Any | None:
    """Resolve and cache the evaluator LLM."""
    if "instance" not in _model_cache:
        from tools.evaluation.evaluators import resolve_model
        _model_cache["instance"] = resolve_model()
    return _model_cache["instance"]


def _parse_json_list(raw: str) -> list[Any] | None:
    """Parse a JSON string into a list, returning None for empty/blank input."""
    if not raw or not raw.strip():
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else None
    except (json.JSONDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_evaluators() -> str:
    """List all available evaluator types with their metadata.

    Returns a JSON array of evaluator descriptors including name,
    description, and required input fields.
    """
    from tools.evaluation.evaluators import REGISTRY

    result = [
        {"type": key, **asdict(info)}
        for key, info in REGISTRY.items()
    ]
    return json.dumps(result, indent=2)


@mcp.tool()
async def evaluate(
    evaluator_type: str,
    user_input: str,
    actual_output: str,
    expected_output: str = "",
    context: str = "",
    retrieval_context: str = "",
    criteria: str = "",
    threshold: float = 0.5,
) -> str:
    """Evaluate an agent's output using a specified metric.

    Args:
        evaluator_type: Evaluator to use (run list_evaluators to see options).
        user_input: The original user query / input.
        actual_output: The agent's response / output to evaluate.
        expected_output: Optional ground-truth / reference answer.
        context: Optional JSON array of context document strings.
        retrieval_context: Optional JSON array of retrieved context strings.
        criteria: Custom evaluation criteria (required for g_eval type).
        threshold: Minimum passing score between 0.0 and 1.0.
    """
    from tools.evaluation.evaluators import run_evaluation

    try:
        result = await run_evaluation(
            evaluator_type,
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output or None,
            context=_parse_json_list(context),
            retrieval_context=_parse_json_list(retrieval_context),
            criteria=criteria or None,
            threshold=threshold,
            model=_get_model(),
        )
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("Evaluation failed for %s", evaluator_type)
        return json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "evaluator_type": evaluator_type,
        })


@mcp.tool()
async def evaluate_trajectory(
    user_input: str,
    actual_output: str,
    tools_called: str,
    expected_tools: str = "",
    threshold: float = 0.5,
) -> str:
    """Evaluate an agent's tool-calling trajectory.

    Scores whether the agent selected the correct tools with correct
    parameters using the deterministic tool_correctness metric.

    Args:
        user_input: The original user query / input.
        actual_output: The agent's final response.
        tools_called: JSON array of tool calls, each with "name" and
            optional "input_parameters" keys.
        expected_tools: Optional JSON array of expected tool calls
            with the same structure.
        threshold: Minimum passing score between 0.0 and 1.0.
    """
    from tools.evaluation.evaluators import run_evaluation

    parsed_called = _parse_json_list(tools_called)
    if not parsed_called:
        return json.dumps({"error": "tools_called must be a non-empty JSON array"})

    try:
        result = await run_evaluation(
            "tool_correctness",
            input=user_input,
            actual_output=actual_output,
            tools_called=parsed_called,
            expected_tools=_parse_json_list(expected_tools),
            threshold=threshold,
            model=_get_model(),
        )
        return json.dumps(result, indent=2)
    except Exception as exc:
        logger.exception("Trajectory evaluation failed")
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})


@mcp.tool()
async def batch_evaluate(
    evaluator_type: str,
    items: str,
    criteria: str = "",
    threshold: float = 0.5,
) -> str:
    """Run the same evaluator across multiple input/output pairs.

    Args:
        evaluator_type: Evaluator to use (run list_evaluators to see options).
        items: JSON array of objects, each with at minimum "input" and
            "actual_output" keys.  May also include "expected_output",
            "context", and "retrieval_context".
        criteria: Custom evaluation criteria (for g_eval type).
        threshold: Minimum passing score between 0.0 and 1.0.
    """
    from tools.evaluation.evaluators import run_evaluation

    parsed_items = _parse_json_list(items)
    if not parsed_items:
        return json.dumps({"error": "items must be a non-empty JSON array"})

    model = _get_model()
    results: list[dict[str, Any]] = []

    for idx, item in enumerate(parsed_items):
        if not isinstance(item, dict):
            results.append({"error": f"Item {idx} is not an object", "index": idx})
            continue
        try:
            result = await run_evaluation(
                evaluator_type,
                input=item.get("input", ""),
                actual_output=item.get("actual_output", ""),
                expected_output=item.get("expected_output"),
                context=item.get("context"),
                retrieval_context=item.get("retrieval_context"),
                criteria=criteria or None,
                threshold=threshold,
                model=model,
            )
            result["index"] = idx
            results.append(result)
        except Exception as exc:
            results.append({
                "error": f"{type(exc).__name__}: {exc}",
                "index": idx,
            })

    summary = {
        "total": len(results),
        "passed": sum(1 for r in results if r.get("success")),
        "failed": sum(1 for r in results if "success" in r and not r["success"]),
        "errors": sum(1 for r in results if "error" in r),
    }
    return json.dumps({"summary": summary, "results": results}, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Agent Evaluator Service MCP Server")
    parser.add_argument(
        "--transport",
        choices=["streamable-http", "stdio"],
        default="streamable-http",
    )
    parser.add_argument("--host", default=_DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    args = parser.parse_args()

    logger.info(
        "Starting Agent Evaluator Service MCP server (transport=%s, host=%s, port=%s)",
        args.transport, args.host, args.port,
    )

    mcp.settings.host = args.host
    mcp.settings.port = args.port

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
