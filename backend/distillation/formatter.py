"""Convert trajectories into mlx_lm ChatDataset JSONL (``messages`` + tools)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def trajectory_to_sft_example(trajectory: dict[str, Any]) -> dict[str, Any]:
    """One mlx_lm chat-SFT row.  The student tokenizer is applied at train time."""
    messages = list(trajectory.get("messages") or [])
    return {
        "messages": messages,
        "session_id": trajectory.get("session_id") or "",
        "tools_used": list(trajectory.get("tools_used") or []),
    }


def render_with_tokenizer(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Apply the student chat template.  Used for round-trip tests and probes."""
    if tokenizer is None:
        raise ValueError("A student tokenizer is required to render SFT text")
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if tools:
        kwargs["tools"] = tools
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("tools", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def assert_round_trip(
    rendered: str,
    family: str,
    expected_names: list[str],
) -> None:
    """Raise ``ValueError`` if *rendered* does not parse back to *expected_names*."""
    from chat_models.mlx._native_tool_parsing import parse_native_tool_calls

    parsed = parse_native_tool_calls(rendered, family) or []
    got = [c.get("name") for c in parsed]
    if got != expected_names:
        raise ValueError(
            f"Student template round-trip failed for family={family!r}: "
            f"expected {expected_names}, parsed {got}"
        )


def to_sft_jsonl(
    trajectories: list[dict[str, Any]],
    output_path: Path,
    *,
    tokenizer: Any | None = None,
    family: str = "qwen",
    verify_round_trip: bool = False,
) -> Path:
    """Write ChatDataset JSONL.  Optionally verify the student template."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for traj in trajectories:
            example = trajectory_to_sft_example(traj)
            if verify_round_trip:
                if tokenizer is None:
                    raise ValueError("verify_round_trip requires a student tokenizer")
                rendered = render_with_tokenizer(example["messages"], tokenizer)
                assert_round_trip(rendered, family, list(example.get("tools_used") or []))
            fh.write(json.dumps(example, default=str) + "\n")
            written += 1
    logger.info("Wrote %d SFT examples to %s", written, output_path)
    return output_path


def qwen_probe_tokenizer() -> Any:
    """Minimal tokenizer-shaped object for tests — emits Qwen ``<tool_call>`` markup.

    Not used in production.  Production rendering goes through the student
    model's real ``apply_chat_template``.
    """

    class _Probe:
        chat_template = "tools <tool_call>"

        def apply_chat_template(
            self,
            messages: list[dict[str, Any]],
            tokenize: bool = False,
            add_generation_prompt: bool = False,
            tools: list | None = None,
            **kwargs: Any,
        ) -> str:
            parts: list[str] = []
            for msg in messages:
                role = msg.get("role") or "user"
                content = msg.get("content") or ""
                parts.append(f"<|im_start|>{role}\n{content}")
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    payload = {
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or {},
                    }
                    parts.append(f"<tool_call>{json.dumps(payload)}</tool_call>")
                parts.append("<|im_end|>")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "\n".join(parts)

    return _Probe()
