"""Unit tests for Playwright text-level error detection (fix #3).

Playwright MCP returns most *action* failures (stale element ``ref``, action
timeout, navigation error) as a SUCCESSFUL tool result (``isError=False``) with
the failure written only into the text body.  Before the fix the loop guard
recorded those as successes, so its failure-loop detector never tripped and a
``browser_click`` on a stale ref could repeat hundreds of times.

``_raise_on_pw_error`` converts such results into a real ``ToolException`` so
that (a) the loop guard counts them as failures and trips after
``max_identical`` repeats, and (b) ``handle_tool_error=True`` still surfaces the
text to the model as a corrective message.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool, ToolException

from tools.loop_guard import (
    ToolLoopDetected,
    build_default_guard,
    wrap_with_loop_guard,
)

from backend.mcp_manager import (
    _PLAYWRIGHT_ERROR_SENTINELS,
    _raise_on_pw_error,
    _tool_result_text,
)

# A realistic Playwright failure result (stale element ref).
_STALE_REF_RESULT = (
    "### Error\n"
    "Error: Ref e127 not found in the current page snapshot. "
    "Try capturing new snapshot."
)

# A realistic *successful* result.  Note it contains the word "errors" in the
# console summary — which must NOT be treated as a failure.
_HEALTHY_RESULT = (
    "### Ran Playwright code\n"
    "```js\nawait page.getByRole('button').click();\n```\n"
    "### Page\n"
    "- Page URL: https://example.com\n"
    "- Console: 1 errors, 9 warnings\n"
    "### Snapshot\n"
    "- button [ref=e42]\n"
)


def _make_tool(name: str, impl):
    return StructuredTool.from_function(
        coroutine=impl, name=name, description=f"{name} tool",
    )


async def test_error_sentinel_result_raises():
    """A result carrying an error sentinel is converted into a ToolException."""
    async def impl(ref: str) -> str:
        return _STALE_REF_RESULT

    tool = _make_tool("browser_click", impl)
    _raise_on_pw_error(tool)

    with pytest.raises(ToolException) as ei:
        await tool.coroutine(ref="e127")
    # The original failure text is preserved for the model's corrective message.
    assert "not found in the current page snapshot" in str(ei.value)


async def test_healthy_result_does_not_raise():
    """A successful result (even one mentioning 'errors' in the console line)
    must pass through untouched."""
    async def impl(ref: str) -> str:
        return _HEALTHY_RESULT

    tool = _make_tool("browser_click", impl)
    _raise_on_pw_error(tool)

    out = await tool.coroutine(ref="e42")
    assert out == _HEALTHY_RESULT


async def test_list_content_blocks_are_inspected():
    """Error detection also works for list-of-content-block results."""
    async def impl(ref: str):
        return [{"type": "text", "text": _STALE_REF_RESULT}]

    tool = _make_tool("browser_click", impl)
    _raise_on_pw_error(tool)

    with pytest.raises(ToolException):
        await tool.coroutine(ref="e127")


async def test_content_and_artifact_tuple_is_inspected():
    """response_format=content_and_artifact tuples are unwrapped for matching."""
    async def impl(ref: str):
        return (_STALE_REF_RESULT, {"some": "artifact"})

    tool = _make_tool("browser_click", impl)
    _raise_on_pw_error(tool)

    with pytest.raises(ToolException):
        await tool.coroutine(ref="e127")


async def test_net_error_sentinel_raises():
    """Navigation network failures (net::ERR_*) are detected too."""
    async def impl(url: str) -> str:
        return "### Result\nnet::ERR_NAME_NOT_RESOLVED at https://nope.invalid"

    tool = _make_tool("browser_navigate", impl)
    _raise_on_pw_error(tool)

    with pytest.raises(ToolException):
        await tool.coroutine(url="https://nope.invalid")


async def test_stale_ref_loop_now_trips_the_guard():
    """End-to-end reproduction of the original bug: a stale-ref click that the
    model repeats must trip the failure-loop guard after ``max_identical``
    repeats — instead of looping forever as it did before the fix.

    This mirrors the real wiring order in ``_load_builtin_playwright``:
    ``_raise_on_pw_error`` is applied *before* ``wrap_with_loop_guard`` so the
    guard's try/except observes the raised ToolException.
    """
    guard = build_default_guard(
        max_identical=3,
        max_identical_success=None,
        max_no_progress=None,
        window=8,
    )

    async def impl(ref: str) -> str:
        return _STALE_REF_RESULT

    tool = _make_tool("browser_click", impl)
    _raise_on_pw_error(tool)
    wrap_with_loop_guard(tool, guard)

    # First three identical failing clicks surface the raw Playwright error
    # (recorded as failures) — but are NOT yet loop-detected.
    for _ in range(3):
        with pytest.raises(ToolException) as ei:
            await tool.coroutine(ref="e127")
        assert not isinstance(ei.value, ToolLoopDetected)

    # The fourth identical click trips the loop guard before the tool runs.
    with pytest.raises(ToolLoopDetected):
        await tool.coroutine(ref="e127")


def test_sentinels_cover_expected_failures():
    """Guard against accidental edits to the sentinel set."""
    assert "### Error" in _PLAYWRIGHT_ERROR_SENTINELS
    assert "not found in the current page snapshot" in _PLAYWRIGHT_ERROR_SENTINELS
    # The bare "errors" console summary must not be a sentinel (false positives).
    assert "errors" not in _PLAYWRIGHT_ERROR_SENTINELS
    assert _tool_result_text("plain") == "plain"
