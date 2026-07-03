"""Live tests: MLX ReAct shim, parametrized per model family.

Each model family (Qwen, DeepSeek-R1, Kimi) emits text in a structurally
different way that the ReAct shim must parse correctly.  Parser tests run on
realistic static strings (no model download required); model-call tests are
gated on env vars.

Marks:
* ``live``      — all tests in this file
* ``mlx_qwen``  — gated on MLX_QWEN_MODEL_ID env var
* ``mlx_deepseek`` — gated on MLX_DEEPSEEK_MODEL_ID env var
* ``mlx_kimi``  — gated on MLX_KIMI_MODEL_ID env var

Run examples::

    pytest -m "live"                          # parser tests only (no model needed)
    pytest -m "live and mlx_qwen"             # Qwen model tests
    pytest -m "live and mlx_deepseek"         # DeepSeek model tests
    pytest -m "live and mlx_kimi"             # Kimi model tests
"""

from __future__ import annotations

import json
import os
import re

import pytest

pytestmark = pytest.mark.live

# ---------------------------------------------------------------------------
# Env-var gates
# ---------------------------------------------------------------------------

_QWEN_MODEL = os.environ.get("MLX_QWEN_MODEL_ID", "")
_DEEPSEEK_MODEL = os.environ.get("MLX_DEEPSEEK_MODEL_ID", "")
_KIMI_MODEL = os.environ.get("MLX_KIMI_MODEL_ID", "")

_require_qwen = pytest.mark.skipif(not _QWEN_MODEL, reason="MLX_QWEN_MODEL_ID not set")
_require_deepseek = pytest.mark.skipif(not _DEEPSEEK_MODEL, reason="MLX_DEEPSEEK_MODEL_ID not set")
_require_kimi = pytest.mark.skipif(not _KIMI_MODEL, reason="MLX_KIMI_MODEL_ID not set")


# ---------------------------------------------------------------------------
# Parser unit tests — no model download required
# These exercise `parse_action` and `strip_think_tags` on realistic strings.
# ---------------------------------------------------------------------------


class TestParser:
    """Core parser logic: these tests run without any model."""

    def test_parse_action_fenced_json(self):
        from middleware._react_core import parse_action

        text = (
            "Thought: I need to look this up.\n"
            "Action:\n"
            "```json\n"
            '{"action": "web_research", "action_input": {"query": "Python release notes"}}\n'
            "```"
        )
        result = parse_action(text)
        assert result is not None, f"parse_action returned None for fenced JSON"
        tool_name, args = result
        assert tool_name == "web_research"
        assert args == {"query": "Python release notes"}

    def test_parse_action_inline_json(self):
        from middleware._react_core import parse_action

        text = 'Action: {"action": "privacy_status", "action_input": {}}'
        result = parse_action(text)
        assert result is not None
        assert result[0] == "privacy_status"

    def test_parse_action_returns_none_for_final_answer(self):
        from middleware._react_core import parse_action

        text = "Final Answer: The capital of France is Paris."
        assert parse_action(text) is None

    def test_parse_action_returns_none_for_plain_text(self):
        from middleware._react_core import parse_action

        assert parse_action("Just a normal response.") is None

    def test_strip_think_tags_closed(self):
        from middleware._react_core import strip_think_tags

        text = "<think>I need to reason here.</think>\nAction: ..."
        cleaned = strip_think_tags(text)
        assert "<think>" not in cleaned
        assert "Action:" in cleaned

    def test_strip_think_tags_unclosed(self):
        """Unclosed <think> tags (truncated at token limit) must also be stripped."""
        from middleware._react_core import strip_think_tags

        text = "<think>Partial reasoning that never closes"
        cleaned = strip_think_tags(text)
        assert "<think>" not in cleaned
        assert cleaned.strip() == ""

    def test_split_final_answer_strips_think_block(self):
        from middleware._react_core import split_final_answer

        text = "<think>Some reasoning.</think>\nFinal Answer: Paris"
        answer, _thought = split_final_answer(text)
        assert "think" not in answer.lower()
        assert "Paris" in answer

    def test_stop_tokens_stripped(self):
        from middleware._react_core import strip_stop_tokens

        text = "Hello<|im_end|> world<|eot_id|>"
        cleaned = strip_stop_tokens(text)
        assert "<|im_end|>" not in cleaned
        assert "<|eot_id|>" not in cleaned
        assert "Hello" in cleaned


# ---------------------------------------------------------------------------
# Qwen / Qwen3 family
# ---------------------------------------------------------------------------


class TestQwenParser:
    """Parser behaviour with realistic Qwen3-family output strings."""

    _QWEN_WITH_THINK = (
        "<think>\n"
        "I should look up the status using the privacy_status tool.\n"
        "</think>\n"
        "Thought: state: need privacy info | next: call privacy_status\n"
        "Action:\n"
        "```json\n"
        '{"action": "privacy_status", "action_input": {}}\n'
        "```\n"
    )

    _QWEN_FINAL = (
        "<think>All done.</think>\n"
        "Final Answer: The privacy lock is currently disengaged."
    )

    def test_qwen_action_extracted_after_think_block(self):
        from middleware._react_core import parse_action

        result = parse_action(self._QWEN_WITH_THINK)
        assert result is not None, "Failed to parse action from Qwen output with <think> block"
        tool_name, _ = result
        assert tool_name == "privacy_status"

    def test_qwen_think_content_not_in_args(self):
        from middleware._react_core import parse_action

        result = parse_action(self._QWEN_WITH_THINK)
        assert result is not None
        _, args = result
        assert "think" not in json.dumps(args).lower()

    def test_qwen_final_answer_parses_correctly(self):
        from middleware._react_core import split_final_answer

        answer, _ = split_final_answer(self._QWEN_FINAL)
        assert "disengaged" in answer.lower()
        assert "<think>" not in answer

    def test_qwen_stop_token_stripped(self):
        from middleware._react_core import strip_stop_tokens

        raw = "Final Answer: Done<|im_end|>"
        cleaned = strip_stop_tokens(raw)
        assert "<|im_end|>" not in cleaned


@pytest.mark.mlx_qwen
@_require_qwen
class TestQwenLive:
    """Live calls to an actual Qwen model — requires MLX_QWEN_MODEL_ID."""

    @pytest.fixture(scope="class")
    def qwen_llm(self):
        import os
        os.environ["HF_LLM_MODEL_ID"] = _QWEN_MODEL
        # Keep thinking OFF: with force_action the wrapper already compels an
        # Action: block, and a <think> preamble would consume the MLX_MAX_TOKENS
        # budget and truncate the action JSON before it closes.
        os.environ["MLX_THINKING"] = "false"
        from deep_agent.model_factory import create_llm
        return create_llm("mlx")

    @pytest.fixture(scope="class")
    def dummy_tool(self):
        from langchain_core.tools import tool

        @tool
        def privacy_status() -> str:
            """Return the current privacy lock status."""
            return "Privacy lock: disengaged"

        return privacy_status

    @pytest.mark.asyncio
    async def test_qwen_react_produces_parseable_action(self, qwen_llm, dummy_tool):
        """A tool-requiring prompt must produce a parseable tool call.

        Uses the ``force_action`` wrapper path — the production mechanism that
        makes models emit an ``Action:`` block instead of fabricating a direct
        answer.  Without it, Qwen simply replies to the question conversationally.
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware.react_wrapper import MLXReActWrapper

        wrapper = MLXReActWrapper(inner=qwen_llm, force_action=True).bind_tools([dummy_tool])
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is my privacy lock status?"),
        ]
        resp = await wrapper.ainvoke(messages)

        assert resp.tool_calls, (
            f"Expected a parseable tool call from Qwen; got content:\n"
            f"{str(resp.content)[:800]}"
        )
        assert resp.tool_calls[0]["name"] == "privacy_status"

    @pytest.mark.asyncio
    async def test_qwen_think_tags_not_in_extracted_args(self, qwen_llm, dummy_tool):
        """Tool call arguments must not contain <think> content."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware.react_wrapper import MLXReActWrapper

        wrapper = MLXReActWrapper(inner=qwen_llm, force_action=True).bind_tools([dummy_tool])
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is my privacy lock status?"),
        ]
        resp = await wrapper.ainvoke(messages)

        if not resp.tool_calls:
            pytest.skip("No tool call produced; skipping args-content check")
        args = resp.tool_calls[0]["args"]
        assert "<think>" not in json.dumps(args), (
            f"<think> tag found in extracted args: {args}"
        )

    @pytest.mark.asyncio
    async def test_qwen_force_action_reduces_preamble(self, qwen_llm, dummy_tool):
        """MLXReActWrapper with force_action=True should avoid long conversational preambles."""
        from langchain_core.tools import tool as lc_tool
        from middleware.react_wrapper import MLXReActWrapper

        wrapper = MLXReActWrapper(inner=qwen_llm, force_action=True).bind_tools([dummy_tool])

        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is my privacy lock status?"),
        ]
        resp = await wrapper.ainvoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        # With force_action, output before Action: should be brief.
        action_idx = text.lower().find("action:")
        preamble = text[:action_idx] if action_idx != -1 else text
        assert len(preamble) < 500, (
            f"force_action preamble is longer than expected ({len(preamble)} chars): "
            f"{preamble[:300]}"
        )


# ---------------------------------------------------------------------------
# DeepSeek R1 / R1-Distill family
# ---------------------------------------------------------------------------


class TestDeepSeekParser:
    """Parser behaviour with realistic DeepSeek-R1-family output strings."""

    _DEEPSEEK_WITH_THINK = (
        "<think>\n"
        "The user wants to know their privacy status. I should call privacy_status.\n"
        "</think>\n"
        "Thought: state: need status | next: call privacy_status\n"
        "Action:\n"
        "```json\n"
        '{"action": "privacy_status", "action_input": {}}\n'
        "```\n"
    )

    _DEEPSEEK_DOUBLE_ACTION = (
        "<think>Let me call the tool.</think>\n"
        "Thought: calling privacy_status\n"
        "Action:\n"
        "```json\n"
        '{"action": "privacy_status", "action_input": {}}\n'
        "```\n"
        "Action:\n"
        "```json\n"
        '{"action": "privacy_status", "action_input": {}}\n'
        "```\n"
    )

    def test_deepseek_action_after_think_block(self):
        from middleware._react_core import parse_action

        result = parse_action(self._DEEPSEEK_WITH_THINK)
        assert result is not None, "Failed to parse action from DeepSeek output"
        assert result[0] == "privacy_status"

    def test_deepseek_think_content_stripped_from_action(self):
        """The <think> block must not appear in the extracted tool name."""
        from middleware._react_core import parse_action

        result = parse_action(self._DEEPSEEK_WITH_THINK)
        assert result is not None
        tool_name, _ = result
        assert "<think>" not in tool_name

    def test_deepseek_first_action_wins_on_double_emit(self):
        """When DeepSeek echoes an Action block, parse_action should return
        exactly one result (the first valid one)."""
        from middleware._react_core import parse_action

        result = parse_action(self._DEEPSEEK_DOUBLE_ACTION)
        assert result is not None
        # Result is a single (name, args) tuple — not a list.
        assert isinstance(result, tuple) and len(result) == 2


@pytest.mark.mlx_deepseek
@_require_deepseek
class TestDeepSeekLive:
    """Live calls to an actual DeepSeek-R1 model — requires MLX_DEEPSEEK_MODEL_ID."""

    @pytest.fixture(scope="class")
    def deepseek_llm(self):
        import os
        os.environ["HF_LLM_MODEL_ID"] = _DEEPSEEK_MODEL
        from deep_agent.model_factory import create_llm
        return create_llm("mlx")

    @pytest.fixture(scope="class")
    def dummy_tool(self):
        from langchain_core.tools import tool

        @tool
        def privacy_status() -> str:
            """Return the current privacy lock status."""
            return "Privacy lock: disengaged"

        return privacy_status

    @pytest.mark.asyncio
    async def test_deepseek_react_action_extracted(self, deepseek_llm, dummy_tool):
        """Model output must contain an Action: block that parse_action can handle."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware._react_core import TOOL_SECTION_TEMPLATE, parse_action, render_tools

        tool_section = TOOL_SECTION_TEMPLATE.format(tool_descriptions=render_tools([dummy_tool]))
        system = SystemMessage(content="You are a helpful assistant." + tool_section)
        human = HumanMessage(content="What is my privacy lock status?")

        resp = await deepseek_llm.ainvoke([system, human])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        result = parse_action(text)
        assert result is not None, (
            f"parse_action returned None for DeepSeek output:\n{text[:800]}"
        )

    @pytest.mark.asyncio
    async def test_deepseek_action_not_inside_think_block(self, deepseek_llm, dummy_tool):
        """The parsed tool name must come from outside the <think> block."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware._react_core import (
            TOOL_SECTION_TEMPLATE,
            parse_action,
            render_tools,
            strip_think_tags,
        )

        tool_section = TOOL_SECTION_TEMPLATE.format(tool_descriptions=render_tools([dummy_tool]))
        system = SystemMessage(content="You are a helpful assistant." + tool_section)
        human = HumanMessage(content="Check my privacy status.")

        resp = await deepseek_llm.ainvoke([system, human])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        text_no_think = strip_think_tags(text)

        result = parse_action(text_no_think)
        assert result is not None, (
            f"No Action: block found outside <think> block:\n{text[:800]}"
        )

    @pytest.mark.asyncio
    async def test_deepseek_tool_result_acknowledged(self, deepseek_llm, dummy_tool):
        """After injecting a tool result, the model must not emit an empty response."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        from middleware._react_core import (
            TOOL_SECTION_TEMPLATE,
            make_tool_call_id,
            parse_action,
            render_tools,
        )

        tool_section = TOOL_SECTION_TEMPLATE.format(tool_descriptions=render_tools([dummy_tool]))
        system = SystemMessage(content="You are a helpful assistant." + tool_section)
        tc_id = make_tool_call_id()
        ai_with_call = AIMessage(
            content="",
            tool_calls=[{"id": tc_id, "name": "privacy_status", "args": {}}],
        )
        tool_result = ToolMessage(content="Privacy lock: disengaged", tool_call_id=tc_id)

        resp = await deepseek_llm.ainvoke([
            system,
            HumanMessage(content="What is my privacy lock status?"),
            ai_with_call,
            tool_result,
        ])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        assert text.strip(), "DeepSeek returned empty response after tool result"


# ---------------------------------------------------------------------------
# Kimi (Moonshot) family
# ---------------------------------------------------------------------------


class TestKimiParser:
    """Parser behaviour with realistic Kimi-family output strings."""

    _KIMI_VERBOSE_PREAMBLE = (
        "Let me think through this step by step.\n"
        "The user is asking about their privacy status, so I should use the privacy_status tool.\n"
        "First, let me understand what information I need...\n"
        "Thought: state: need privacy info | next: call privacy_status\n"
        "Action:\n"
        "```json\n"
        '{"action": "privacy_status", "action_input": {}}\n'
        "```\n"
    )

    _KIMI_TRAILING_COMMA = (
        "Action:\n"
        "```json\n"
        '{"action": "privacy_status", "action_input": {"extra": "value",}}\n'
        "```\n"
    )

    def test_kimi_action_found_after_preamble(self):
        from middleware._react_core import parse_action

        result = parse_action(self._KIMI_VERBOSE_PREAMBLE)
        assert result is not None, (
            "parse_action failed on Kimi output with verbose preamble"
        )
        assert result[0] == "privacy_status"

    def test_kimi_tool_name_not_polluted_by_preamble(self):
        from middleware._react_core import parse_action

        result = parse_action(self._KIMI_VERBOSE_PREAMBLE)
        assert result is not None
        tool_name, _ = result
        assert len(tool_name) < 100, f"Tool name suspiciously long: {tool_name!r}"
        assert "\n" not in tool_name, f"Newline in tool name: {tool_name!r}"


@pytest.mark.mlx_kimi
@_require_kimi
class TestKimiLive:
    """Live calls to an actual Kimi model — requires MLX_KIMI_MODEL_ID."""

    @pytest.fixture(scope="class")
    def kimi_llm(self):
        import os
        os.environ["HF_LLM_MODEL_ID"] = _KIMI_MODEL
        from deep_agent.model_factory import create_llm
        return create_llm("mlx")

    @pytest.fixture(scope="class")
    def dummy_tool(self):
        from langchain_core.tools import tool

        @tool
        def privacy_status() -> str:
            """Return the current privacy lock status."""
            return "Privacy lock: disengaged"

        return privacy_status

    @pytest.mark.asyncio
    async def test_kimi_react_action_found_after_preamble(self, kimi_llm, dummy_tool):
        """parse_action must find Action: even when Kimi emits a verbose preamble."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware._react_core import TOOL_SECTION_TEMPLATE, parse_action, render_tools

        tool_section = TOOL_SECTION_TEMPLATE.format(tool_descriptions=render_tools([dummy_tool]))
        system = SystemMessage(content="You are a helpful assistant." + tool_section)
        human = HumanMessage(content="What is my privacy lock status?")

        resp = await kimi_llm.ainvoke([system, human])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        result = parse_action(text)
        assert result is not None, (
            f"parse_action returned None for Kimi output:\n{text[:800]}"
        )

    @pytest.mark.asyncio
    async def test_kimi_action_args_are_valid_json_shape(self, kimi_llm, dummy_tool):
        """The args extracted from Kimi's Action block must be a dict (valid JSON shape)."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware._react_core import TOOL_SECTION_TEMPLATE, parse_action, render_tools

        tool_section = TOOL_SECTION_TEMPLATE.format(tool_descriptions=render_tools([dummy_tool]))
        system = SystemMessage(content="You are a helpful assistant." + tool_section)
        human = HumanMessage(content="Check privacy status.")

        resp = await kimi_llm.ainvoke([system, human])
        text = resp.content if isinstance(resp.content, str) else str(resp.content)

        result = parse_action(text)
        if result is None:
            pytest.skip("No action block found")
        _, args = result
        assert isinstance(args, dict), f"Expected dict args; got {type(args)}: {args}"

    @pytest.mark.asyncio
    async def test_kimi_force_action_reduces_preamble(self, kimi_llm, dummy_tool):
        """MLXReActWrapper with force_action=True should shorten Kimi's preamble."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from middleware.react_wrapper import MLXReActWrapper

        wrapper = MLXReActWrapper(inner=kimi_llm, force_action=True).bind_tools([dummy_tool])
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="What is my privacy lock status?"),
        ]
        resp = await wrapper.ainvoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        action_idx = text.lower().find("action:")
        preamble = text[:action_idx] if action_idx != -1 else text
        assert len(preamble) < 800, (
            f"force_action preamble is longer than expected ({len(preamble)} chars)"
        )


# ---------------------------------------------------------------------------
# Cross-family: tool schema is identical regardless of which model is loaded
# ---------------------------------------------------------------------------


def test_react_tool_schema_identical_across_render_calls():
    """render_tools should produce the same string for the same tool list
    regardless of when it's called (no global state)."""
    from langchain_core.tools import tool
    from middleware._react_core import render_tools

    @tool
    def sample_tool(query: str) -> str:
        """A sample tool for testing."""
        return query

    render1 = render_tools([sample_tool])
    render2 = render_tools([sample_tool])

    assert render1 == render2, "render_tools produced different output on consecutive calls"
    assert "sample_tool" in render1
