"""Small-context middleware must count and shrink the OpenAI tools array."""

from __future__ import annotations

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage, SystemMessage

from middleware.context_truncation import SmallContextTruncationMiddleware


def _fat_tool(i: int) -> dict:
    props = {f"field_{j}": {"type": "string", "description": "x" * 80} for j in range(40)}
    return {
        "type": "function",
        "function": {
            "name": f"bulk_tool_{i}",
            "description": "A very long MCP tool description. " + ("more " * 40),
            "parameters": {"type": "object", "properties": props},
        },
    }


def test_truncation_compacts_tools_to_fit_40k_class_budget():
    tools = [_fat_tool(i) for i in range(40)]
    req = ModelRequest(
        model=object(),
        messages=[HumanMessage(content="hello")],
        system_message=SystemMessage(content="You are Otto."),
        tools=tools,
    )
    mw = SmallContextTruncationMiddleware(max_input_tokens=30720, safety_margin_tokens=256)
    fitted = mw._fit(req)
    assert fitted is not req
    assert len(fitted.tools) <= len(tools)
    for t in fitted.tools:
        schema = t.get("function", {}).get("parameters", {})
        assert schema.get("properties") == {}
