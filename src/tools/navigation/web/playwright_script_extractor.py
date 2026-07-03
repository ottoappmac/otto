"""Extract Playwright code blocks from a DeepAgent conversation and assemble output files.

After a Playwright MCP browser-automation run, the conversation history contains
``### Ran Playwright code`` blocks in every ``ToolMessage``.  This module splices
those blocks into a runnable ``.spec.ts`` file and generates a companion
``package.json``, removing the need for the LLM to ``write_file`` large content
(which small MLX models struggle with due to JSON nesting-depth errors).

Usage from a notebook::

    from tools.navigation.web.playwright_script_extractor import extract_playwright_outputs
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

_PW_CODE_RE = re.compile(
    r"### Ran Playwright code\s*```(?:js|javascript)?\s*(.*?)```",
    re.DOTALL,
)

_ORDER_NUMBER_RE = re.compile(
    r"(?:order|sales order|order number)[^\d]*(\d{4,6})",
    re.IGNORECASE,
)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def _extract_events(messages: list) -> list[tuple]:
    """Extract code blocks and tab-switch events from the conversation.

    Returns a list of event tuples in chronological order:

    - ``("code", tool_name, code_str)`` for Playwright code blocks
    - ``("tab_switch", index)`` for ``browser_tabs`` select actions

    Tab-switch events are detected from ``AIMessage.tool_calls`` so that
    :func:`_build_spec_ts` can emit the correct ``waitForEvent('page')``
    pattern in the generated spec.
    """
    events: list[tuple] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in getattr(msg, "tool_calls", []):
                if tc.get("name") == "browser_tabs":
                    args = tc.get("args", {})
                    if args.get("action") == "select":
                        idx = args.get("index", 0)
                        if idx > 0:
                            events.append(("tab_switch", idx))
        elif isinstance(msg, ToolMessage):
            text = _content_text(msg.content)
            tool_name = getattr(msg, "name", None) or "browser_action"
            for m in _PW_CODE_RE.finditer(text):
                code = m.group(1).strip()
                if code:
                    events.append(("code", tool_name, code))
    return events


def _extract_code_blocks(messages: list) -> list[tuple[str, str]]:
    """Pull all ``### Ran Playwright code`` JS snippets from ToolMessages.

    Returns a list of ``(tool_name, code)`` tuples.  Convenience wrapper
    around :func:`_extract_events` that filters out non-code events.
    """
    return [(e[1], e[2]) for e in _extract_events(messages) if e[0] == "code"]


def _extract_order_number(messages: list) -> str | None:
    """Try to find a sales-order number in the last few AI messages."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        text = _content_text(msg.content)
        m = _ORDER_NUMBER_RE.search(text)
        if m:
            return m.group(1)
    return None


def _build_spec_ts(
    events: list[tuple],
    test_name: str,
    url: str | None = None,
) -> str:
    """Assemble extracted events into a Playwright .spec.ts file.

    Handles ``tab_switch`` events by wrapping the preceding ``.click()``
    in a ``Promise.all([context.waitForEvent('page'), click])`` pattern
    and reassigning ``page`` to the new tab.
    """
    has_tab_switch = any(e[0] == "tab_switch" for e in events)

    if has_tab_switch:
        lines = [
            "import { test, expect } from '@playwright/test';",
            "",
            f"test('{test_name}', async ({{ page: initialPage }}) => {{",
            "  let page = initialPage;",
        ]
    else:
        lines = [
            "import { test, expect } from '@playwright/test';",
            "",
            f"test('{test_name}', async ({{ page }}) => {{",
        ]

    if url:
        lines.append(f"  // Target: {url}")
        lines.append("")

    tab_n = 0
    is_first_block = True
    i = 0
    while i < len(events):
        event = events[i]
        next_is_tab = (
            i + 1 < len(events) and events[i + 1][0] == "tab_switch"
        )

        if event[0] == "code":
            _, tool_name, code = event
            if not is_first_block:
                lines.append("")
            is_first_block = False

            if next_is_tab:
                tab_n += 1
                tab_var = f"newTab{tab_n}" if tab_n > 1 else "newTab"
                lines.append(f"  // {tool_name} — opens new tab")
                code_lines = code.splitlines()
                click_idx = None
                for j in range(len(code_lines) - 1, -1, -1):
                    if ".click()" in code_lines[j]:
                        click_idx = j
                        break

                if click_idx is not None:
                    for line in code_lines[:click_idx]:
                        lines.append(f"  {line}")
                    click_line = code_lines[click_idx].strip()
                    if click_line.startswith("await "):
                        click_line = click_line[6:]
                    click_line = click_line.rstrip(";")
                    lines.append(f"  const [{tab_var}] = await Promise.all([")
                    lines.append(f"    page.context().waitForEvent('page'),")
                    lines.append(f"    {click_line},")
                    lines.append(f"  ]);")
                    lines.append(f"  await {tab_var}.waitForLoadState();")
                    lines.append(f"  page = {tab_var};")
                    for line in code_lines[click_idx + 1:]:
                        lines.append(f"  {line}")
                else:
                    for line in code.splitlines():
                        lines.append(f"  {line}")
                    lines.append("  await page.waitForTimeout(1000);")
                i += 2
                continue
            else:
                lines.append(f"  // {tool_name}")
                for line in code.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("//") or stripped == "":
                        lines.append(f"  {stripped}")
                    else:
                        lines.append(f"  {line}")
                lines.append("  await page.waitForTimeout(1000);")

        i += 1

    lines.append("});")
    lines.append("")
    return "\n".join(lines)


def _build_skill_md(
    blocks: list[tuple[str, str]],
    test_name: str,
    url: str | None = None,
    order_number: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Build a skill markdown file from the extracted data."""
    meta = extra_metadata or {}
    skill_name = meta.get("name", test_name.lower().replace(" ", "-"))
    description = meta.get("description", f"Automate: {test_name}")
    allowed_tools = meta.get("allowed_tools", [
        "browser_navigate", "browser_fill_form", "browser_click",
        "browser_press_key", "browser_snapshot", "browser_tabs",
        "browser_wait_for",
    ])

    tools_yaml = "\n".join(f"  - {t}" for t in allowed_tools)
    sections = [
        "---",
        f"name: {skill_name}",
        f"description: {description}",
        f"allowed-tools:\n{tools_yaml}",
        "---",
        "",
        f"# {test_name}",
        "",
    ]

    if url:
        sections.append(f"**URL**: {url}")
        sections.append("")

    sections.append("## Playwright Steps")
    sections.append("")
    for i, (tool_name, code) in enumerate(blocks, 1):
        sections.append(f"### Step {i}: {tool_name}")
        sections.append("")
        sections.append("```js")
        sections.append(code)
        sections.append("```")
        sections.append("")

    if order_number:
        sections.append("## Result")
        sections.append("")
        sections.append(f"- Sales Order Number: {order_number}")
        sections.append("")

    sections.append("## Gotchas")
    sections.append("")
    sections.append("- Element refs go stale after any click, submit, keypress, or navigation")
    sections.append("- Always call `browser_snapshot()` for fresh refs before the next interaction")
    sections.append("- Use `browser_fill_form` for batch field fills")
    sections.append("- After tab switch, `browser_wait_for(time=3)` before snapshot")
    sections.append("")

    return "\n".join(sections)


PACKAGE_JSON = textwrap.dedent("""\
    {
      "private": true,
      "scripts": {
        "test": "playwright test"
      },
      "devDependencies": {
        "@playwright/test": "^1.50.0"
      }
    }
""")


def extract_playwright_outputs(
    agent: Any,
    thread_id: str,
    output_dir: str | Path,
    test_name: str = "browser automation test",
    url: str | None = None,
    skill_metadata: dict[str, Any] | None = None,
) -> dict[Path, int]:
    """Extract Playwright code from the agent's conversation and write output files.

    Args:
        agent: A ``DeepAgent`` instance (needs ``agent.graph`` for state access).
        thread_id: The thread ID used in the agent run.
        output_dir: Directory to write the output files into.
        test_name: Human-readable test name for the spec file.
        url: The target URL (included as a comment in the spec).
        skill_metadata: Optional dict with ``name``, ``description``,
            ``allowed_tools`` overrides for the skill frontmatter.

    Returns:
        Dict mapping each written file path to its size in bytes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
    messages = state.values.get("messages", [])

    events = _extract_events(messages)
    blocks = [(e[1], e[2]) for e in events if e[0] == "code"]
    order_number = _extract_order_number(messages)

    written: dict[Path, int] = {}

    # 1. spec.ts
    spec_path = output_dir / f"{test_name.lower().replace(' ', '_')}.spec.ts"
    spec_content = _build_spec_ts(events, test_name, url)
    spec_path.write_text(spec_content)
    written[spec_path] = len(spec_content)

    # 2. package.json
    pkg_path = output_dir / "package.json"
    pkg_path.write_text(PACKAGE_JSON)
    written[pkg_path] = len(PACKAGE_JSON)

    # 3. skill.md
    skill_path = output_dir / f"{test_name.lower().replace(' ', '_')}_skill.md"
    skill_content = _build_skill_md(
        blocks, test_name, url, order_number, skill_metadata,
    )
    skill_path.write_text(skill_content)
    written[skill_path] = len(skill_content)

    print(f"\nExtracted {len(blocks)} Playwright code blocks from {len(messages)} messages")
    if order_number:
        print(f"Detected order number: {order_number}")
    print("Output files:")
    for path, size in written.items():
        print(f"  {path.name}: {size:,} bytes")

    return written
