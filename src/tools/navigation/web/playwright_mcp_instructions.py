"""Reusable Playwright MCP instructions for browser automation agents.

Contains general-purpose rules for agents using the Playwright MCP
accessibility-snapshot-based browser tools.

Usage::

    from tools.navigation.web.playwright_mcp_instructions import (
        PLAYWRIGHT_MCP_INSTRUCTIONS,
        build_playwright_prompt,
    )

    prompt = f"... task preamble ... {PLAYWRIGHT_MCP_INSTRUCTIONS}"
"""

from __future__ import annotations

PLAYWRIGHT_MCP_INSTRUCTIONS: str = """\
## Browser Tool Rules

- Do NOT call `write_todos`. Proceed directly to tool calls.
- Follow ONLY the steps given. Do NOT invent extra steps, hallucinate data, or guess field values.
  When all listed steps are done, write outputs and STOP.
- Use `browser_fill_form` with a `fields` array to fill multiple fields in ONE call.
- For single text inputs (search boxes, URL bars, chat inputs, fields that submit on Enter),
  call `browser_type(ref=..., text=..., submit=true)` so the Enter key is pressed in the
  SAME tool call. Only omit `submit=true` when the user explicitly said not to press Enter,
  when the value must be verified by snapshot before submitting, or when the field is part
  of a multi-field form with a separate submit button (use `browser_fill_form` + click).
- Before Save/Submit: `browser_snapshot()` to verify field values match expected data.
- `write_file`: pass `file_path` EXACTLY as given. Do NOT prepend cwd or system paths.

### `ref` argument rules (CRITICAL)

The `ref` argument for browser tools (`browser_click`, `browser_type`,
`browser_fill_form`, `browser_hover`, `browser_select_option`, …) MUST be
a literal element id taken from the most recent accessibility snapshot,
such as `"e23"` or `"e147"`.

It is NOT a CSS selector, NOT an XPath, and NOT a role expression.
Values like `"[role='combobox']"`, `"#search"`, `".btn-primary"`, or
`"input[name=q]"` will be rejected by the MCP server.

Correct:   {"ref": "e23", "text": "hammer"}
Incorrect: {"ref": "[role='combobox']", "text": "hammer"}

If you don't have a current snapshot id, call `browser_snapshot()` first
and pick the id from the returned tree. If the same tool call fails twice
with the same `ref`, STOP retrying that ref — re-snapshot and pick a
different element, or use a different tool (e.g. `browser_press_key`).
"""


def build_playwright_prompt(
    *,
    url: str,
    task_instructions: str,
    output_sections: str = "",
) -> str:
    """Build a complete Playwright MCP agent prompt.

    Args:
        url: The starting URL to navigate to.
        task_instructions: Task-specific execution steps.
        output_sections: Optional output file instructions.

    Returns:
        Formatted prompt string ready for ``agent.astream()``.
    """
    parts = [
        task_instructions.rstrip(),
        "",
        PLAYWRIGHT_MCP_INSTRUCTIONS.rstrip(),
        "",
        "## Execution Start",
        "",
        f"Navigate to: {url}",
    ]
    if output_sections:
        parts += ["", "---", "", output_sections.rstrip()]
    return "\n".join(parts)
