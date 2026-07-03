"""Test PlaywrightSnapshotPruningMiddleware with real LangSmith trace data.

Run:  python -m pytest tests/test_playwright_pruning.py -v -s
  or: python tests/test_playwright_pruning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from middleware.playwright_pruning import _content_text, _prune


def _build_messages_from_langsmith(data: dict) -> list:
    """Reconstruct LangChain message objects from LangSmith JSON export."""
    raw = data.get("messages") or data.get("inputs", {}).get("messages", [])
    msgs = []
    for m in raw:
        t = m["type"]
        content = m["content"]
        if t == "human":
            msgs.append(HumanMessage(content=content, id=m.get("id")))
        elif t == "ai":
            msgs.append(AIMessage(
                content=content,
                tool_calls=m.get("tool_calls", []),
                id=m.get("id"),
            ))
        elif t == "tool":
            msgs.append(ToolMessage(
                content=content,
                name=m.get("name", ""),
                tool_call_id=m.get("tool_call_id", ""),
                status=m.get("status", "success"),
                id=m.get("id"),
            ))
    return msgs


SAMPLE_TRACE = {
    "messages": [
        {
            "type": "human",
            "content": "Navigate to URL, log in, click tile.",
            "id": "h1",
        },
        {
            "type": "ai",
            "content": "Thought: navigate\nAction:\n```json\n{\"action\": \"browser_navigate\", \"action_input\": {\"url\": \"https://example.com\"}}\n```",
            "tool_calls": [{"name": "browser_navigate", "args": {"url": "https://example.com"}, "id": "tc1", "type": "tool_call"}],
            "id": "ai1",
        },
        {
            "type": "tool",
            "content": [{"type": "text", "text": "### Ran Playwright code\n```js\nawait page.goto('https://example.com');\n```\n### Page\n- Page URL: https://example.com\n- Page Title: Login\n### Snapshot\n```yaml\n- textbox \"User\" [ref=e9]\n- textbox \"Password\" [ref=e12]\n- button \"Log On\" [ref=e21]\n```\n### Events\n- console log entry 1\n- console log entry 2"}],
            "name": "browser_navigate",
            "tool_call_id": "tc1",
            "status": "success",
            "id": "t1",
        },
        {
            "type": "ai",
            "content": "Thought: fill form\nAction:\n```json\n{\"action\": \"browser_fill_form\", \"action_input\": {\"fields\": [{\"ref\": \"e9\", \"value\": \"User03\"}]}}\n```",
            "tool_calls": [{"name": "browser_fill_form", "args": {"fields": [{"ref": "e9", "value": "User03"}]}, "id": "tc2", "type": "tool_call"}],
            "id": "ai2",
        },
        {
            "type": "tool",
            "content": [{"type": "text", "text": "### Ran Playwright code\n```js\nawait page.getByRole('textbox', { name: 'User' }).fill('User03');\n```"}],
            "name": "browser_fill_form",
            "tool_call_id": "tc2",
            "status": "success",
            "id": "t2",
        },
        {
            "type": "ai",
            "content": "Thought: click log on\nAction:\n```json\n{\"action\": \"browser_click\", \"action_input\": {\"element\": \"Log On\", \"ref\": \"e21\"}}\n```",
            "tool_calls": [{"name": "browser_click", "args": {"element": "Log On", "ref": "e21"}, "id": "tc3", "type": "tool_call"}],
            "id": "ai3",
        },
        {
            "type": "tool",
            "content": [{"type": "text", "text": "### Ran Playwright code\n```js\nawait page.getByRole('button', { name: 'Log On' }).click();\n```\n### Page\n- Page URL: https://example.com/home\n- Page Title: Home\n### Snapshot\n```yaml\n- button \"Create Sales Orders VA01\" [ref=e56]\n- button \"Change Sales Orders\" [ref=e60]\n```\n### Events\n- navigation complete\n- [ERROR] session timeout"}],
            "name": "browser_click",
            "tool_call_id": "tc3",
            "status": "success",
            "id": "t3",
        },
        {
            "type": "ai",
            "content": "Thought: click tile\nAction:\n```json\n{\"action\": \"browser_click\", \"action_input\": {\"element\": \"Create Sales Orders\", \"ref\": \"e56\"}}\n```",
            "tool_calls": [{"name": "browser_click", "args": {"element": "Create Sales Orders", "ref": "e56"}, "id": "tc4", "type": "tool_call"}],
            "id": "ai4",
        },
        {
            "type": "tool",
            "content": [{"type": "text", "text": "### Ran Playwright code\n```js\nawait page.getByRole('button', { name: 'Create Sales Orders' }).click();\n```\n### Open tabs\n- 0: (current) Home\n- 1: Create Sales Orders\n### Page\n- Page URL: https://example.com/home\n### Snapshot\n```yaml\n\n```\n### Events\n- new tab opened"}],
            "name": "browser_click",
            "tool_call_id": "tc4",
            "status": "success",
            "id": "t4",
        },
        {
            "type": "ai",
            "content": "Thought: switch tab\nAction:\n```json\n{\"action\": \"browser_tabs\", \"action_input\": {\"action\": \"select\", \"index\": 1}}\n```",
            "tool_calls": [{"name": "browser_tabs", "args": {"action": "select", "index": 1}, "id": "tc5", "type": "tool_call"}],
            "id": "ai5",
        },
        {
            "type": "tool",
            "content": [{"type": "text", "text": "### Result\n- 0: Home\n- 1: (current) Create Sales Orders\n### Events\n- [INFO] tab switched\n- [ERROR] session timeout"}],
            "name": "browser_tabs",
            "tool_call_id": "tc5",
            "status": "success",
            "id": "t5",
        },
    ]
}


def _tool_content_chars(msgs: list) -> int:
    return sum(len(_content_text(m.content)) for m in msgs if isinstance(m, ToolMessage))


def _print_tool_msgs(msgs: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for i, m in enumerate(msgs):
        if isinstance(m, ToolMessage):
            text = _content_text(m.content)
            print(f"  [{i}] {m.name} ({len(text)} chars): {text[:120]}{'...' if len(text) > 120 else ''}")


def test_pruning_with_sample():
    msgs = _build_messages_from_langsmith(SAMPLE_TRACE)
    assert len(msgs) == 11, f"Expected 11 messages, got {len(msgs)}"

    before_chars = _tool_content_chars(msgs)
    _print_tool_msgs(msgs, "BEFORE pruning")

    pruned = _prune(msgs, max_messages=40)
    after_chars = _tool_content_chars(pruned)
    _print_tool_msgs(pruned, "AFTER pruning")

    reduction = (1 - after_chars / before_chars) * 100 if before_chars else 0
    print(f"\nTool content: {before_chars} → {after_chars} chars ({reduction:.0f}% reduction)")
    print(f"Message count: {len(msgs)} → {len(pruned)}")

    tool_msgs_after = [m for m in pruned if isinstance(m, ToolMessage)]

    # The LAST snapshot (browser_click t4 — empty yaml) should be kept intact
    t4 = next(m for m in pruned if isinstance(m, ToolMessage) and getattr(m, "name", "") == "browser_click" and m.tool_call_id == "tc4")
    assert "### Snapshot" in _content_text(t4.content), "Latest snapshot should be kept intact"

    # Older snapshot (browser_navigate t1) should be compacted
    t1 = next(m for m in pruned if isinstance(m, ToolMessage) and m.tool_call_id == "tc1")
    t1_text = _content_text(t1.content)
    assert "### Snapshot" not in t1_text, f"Old navigate snapshot should be compacted, got: {t1_text[:200]}"
    assert "### Ran Playwright code" in t1_text, "Old navigate should keep PW code"
    assert "### Page" not in t1_text, "Old navigate should NOT have ### Page"
    assert "### Events" not in t1_text, "Old navigate should NOT have ### Events"

    # browser_fill_form (t2) has no snapshot/page/events, should pass through
    t2 = next(m for m in pruned if isinstance(m, ToolMessage) and m.tool_call_id == "tc2")
    assert "### Ran Playwright code" in _content_text(t2.content)

    # browser_click (t3) with snapshot should be compacted
    t3 = next(m for m in pruned if isinstance(m, ToolMessage) and m.tool_call_id == "tc3")
    t3_text = _content_text(t3.content)
    assert "### Snapshot" not in t3_text, "Old click snapshot should be compacted"
    assert "### Ran Playwright code" in t3_text, "Old click should keep PW code"

    # browser_tabs (t5) has ### Result + ### Events but no snapshot — should be compacted
    t5 = next(m for m in pruned if isinstance(m, ToolMessage) and m.tool_call_id == "tc5")
    t5_text = _content_text(t5.content)
    assert "### Result" not in t5_text, f"browser_tabs ### Result should be compacted, got: {t5_text}"
    assert "### Events" not in t5_text, "browser_tabs ### Events should be compacted"

    assert reduction > 40, f"Expected >40% reduction, got {reduction:.0f}%"
    print("\n✓ All assertions passed!")


def test_pruning_with_json_file():
    """Load a real LangSmith export if available."""
    json_path = Path(__file__).parent.parent / "doc" / "examples" / "deep_agent_orchestration"
    candidates = sorted(json_path.glob("run-*.json")) + sorted(Path.home().glob("Downloads/run-*.json"))
    if not candidates:
        print("No run-*.json files found, skipping real trace test")
        return

    path = candidates[-1]
    print(f"\nLoading real trace: {path.name}")
    data = json.loads(path.read_text())

    raw_msgs = _extract_messages(data)
    if not raw_msgs:
        print("  No messages found in JSON")
        return

    msgs = _build_messages_from_langsmith({"messages": raw_msgs})
    before_chars = _tool_content_chars(msgs)
    _print_tool_msgs(msgs, f"BEFORE pruning ({path.name})")

    pruned = _prune(msgs, max_messages=40)
    after_chars = _tool_content_chars(pruned)
    _print_tool_msgs(pruned, f"AFTER pruning ({path.name})")

    reduction = (1 - after_chars / before_chars) * 100 if before_chars else 0
    print(f"\nTool content: {before_chars} → {after_chars} chars ({reduction:.0f}% reduction)")
    print(f"Message count: {len(msgs)} → {len(pruned)}")


def _extract_messages(data: dict) -> list:
    """Try multiple JSON structures used by LangSmith exports."""
    for key_path in [
        lambda d: d.get("messages"),
        lambda d: d.get("inputs", {}).get("messages"),
        lambda d: d.get("outputs", {}).get("messages"),
    ]:
        msgs = key_path(data)
        if msgs and len(msgs) > 1:
            return msgs
    return []


if __name__ == "__main__":
    test_pruning_with_sample()
    print("\n" + "="*60)
    test_pruning_with_json_file()
