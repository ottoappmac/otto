"""Shared ReAct utilities used by both MLXReActMiddleware and MLXReActWrapper.

This module contains the prompt template, compiled regexes, and pure helper
functions that implement the text-based ReAct shim for text-only LLMs.  It is
an internal implementation detail; import from ``middleware.react_middleware`` or
``middleware.react_wrapper`` instead.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.tools.render import render_text_description_and_args

# ---------------------------------------------------------------------------
# Prompt template injected into the system message
# ---------------------------------------------------------------------------

TOOL_SECTION_TEMPLATE: str = """\

## Available Tools

You have access to the following tools. Call them to gather information or \
perform actions before giving your final answer.

{tool_descriptions}

## Planning Multi-Step Work

If a `write_todos` tool is listed above and the task needs 3 or more steps, make \
your VERY FIRST action a `write_todos` call that lists every step as its own todo \
(set the first to "in_progress", the rest to "pending"), e.g.:

Thought: state: task has 4 steps | next: record the plan
Action:
```json
{{"action": "write_todos", "action_input": {{"todos": [{{"content": "step one", "status": "in_progress"}}, {{"content": "step two", "status": "pending"}}]}}}}
```

On each later turn, take the next step's action and keep the list current (mark the \
finished todo "completed", set the next "in_progress"). For simple 1-2 step tasks, \
skip planning and act directly.

## Response Format

At each step you MUST choose **exactly one** of the two response formats below.

**To call a tool:**

Thought: <one line — state: <what_i_know> | next: <tool_call_reason>
Action:
```json
{{"action": "<tool_name>", "action_input": {{"<param_name>": "<value>"}}}}
```

**To give the final answer (no more tools needed):**

Thought: done — <one-line summary of what was accomplished>
Final Answer: <your complete response to the user>

Keep Thought: to a single concise line. Use pseudocode style, e.g.:
  "state: app=running, controls=loaded | next: type into field[3]"
  "state: calc opened | next: press '1','2','+','3','4','='"
  "state: result=46 | next: close app"

IMPORTANT: After writing the `Action:` block, stop immediately. \
Do NOT continue until you receive the Observation from the tool.
"""

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

# Action block with an explicit ```json ... ``` fence (preferred format)
_ACTION_FENCED_RE = re.compile(
    r"Action:\s*```(?:json)?\s*(\{[\s\S]*?\})\s*```",
    re.IGNORECASE,
)

# Inline JSON after "Action:" without a code fence (fallback)
_ACTION_INLINE_RE = re.compile(
    r"Action:\s*(\{[^`]*\})",
    re.IGNORECASE,
)

# Thinking-model preamble tags (Qwen3, DeepSeek-R1)
_THINK_CLOSED_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>[\s\S]*", re.IGNORECASE)

# Thought line(s) — text between "Thought:" and the next Action:/Final Answer:
_THOUGHT_RE = re.compile(
    r"Thought:\s*(.+?)(?=\n(?:Action:|Final\s+Answer:)|$)",
    re.DOTALL | re.IGNORECASE,
)

# Chat-template stop / control tokens that sometimes leak through into the
# rendered text (Qwen <|im_end|>, Llama <|eot_id|>, DeepSeek <|endoftext|>, etc.).
# Stripping them keeps the user-visible AIMessage clean and prevents future
# turns from echoing template scaffolding back at the model.
_STOP_TOKEN_RE = re.compile(
    r"<\|(?:im_(?:end|start)|eot_id|end_of_text|endoftext|begin_of_text|"
    r"start_header_id|end_header_id)\|>|</s>|<s>",
    re.IGNORECASE,
)

# Final Answer body — everything after the LAST "Final Answer:" marker.
# Uses rfind logic in _split_final_answer rather than a regex so we don't
# accidentally pick up an earlier "Final Answer:" the model wrote inside
# its <think> preamble.


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def extract_thought(text: str) -> str | None:
    """Extract the ``Thought:`` reasoning from a ReAct model response.

    Strips ``<think>`` blocks first so reasoning-model preambles (Qwen3,
    DeepSeek-R1) do not pollute the returned thought text.

    Args:
        text: Raw ``AIMessage.content`` string from the model.

    Returns:
        The thought text (stripped), or ``None`` if no ``Thought:`` line
        was found.
    """
    clean = strip_think_tags(text)
    m = _THOUGHT_RE.search(clean)
    return m.group(1).strip() if m else None


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks emitted by reasoning models.

    Also strips unclosed ``<think>`` tags that appear when a model runs out of
    token budget mid-thought and never emits the closing tag.
    """
    text = _THINK_CLOSED_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def strip_stop_tokens(text: str) -> str:
    """Remove chat-template control tokens (``<|im_end|>`` etc.) from *text*.

    Different model families leak different tokens into the decoded output:
    Qwen uses ``<|im_end|>``, Llama uses ``<|eot_id|>``, DeepSeek uses
    ``<|endoftext|>``.  Cleaning these out keeps the message readable and
    prevents the next turn's prompt from echoing template scaffolding back
    at the model.
    """
    return _STOP_TOKEN_RE.sub("", text)


def split_final_answer(text: str) -> tuple[str, str | None]:
    """Split a ReAct response into ``(answer, thought)``.

    Strips ``<think>`` blocks and chat-template control tokens, extracts the
    ``Thought:`` reasoning, and returns the body that comes after the LAST
    ``Final Answer:`` marker.  When no ``Final Answer:`` marker is present
    the whole cleaned text is returned as the answer (this handles models
    that just output a plain reply without ReAct scaffolding).

    Args:
        text: Raw ``AIMessage.content`` string from the model.

    Returns:
        A two-tuple ``(answer, thought)`` where ``answer`` is the
        user-visible reply (always a string) and ``thought`` is the
        extracted reasoning (or ``None`` when no ``Thought:`` line exists).
    """
    cleaned = strip_stop_tokens(strip_think_tags(text)).strip()
    thought = extract_thought(cleaned)

    # rfind on lowercase copy to find the last "Final Answer:" while keeping
    # the original casing of the body we slice out.
    marker = "final answer:"
    idx = cleaned.lower().rfind(marker)
    if idx != -1:
        answer = cleaned[idx + len(marker):].strip()
    else:
        answer = cleaned

    return answer, thought


def content_to_text(content: str | list[Any]) -> str:
    """Extract plain text from a message's ``content`` field.

    LangChain messages may carry content as a plain ``str`` or as a list of
    content blocks (e.g. ``[{"type": "text", "text": "..."}, ...]``).  This
    helper normalises both forms into a single string.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def render_tools(tools: list[BaseTool | dict[str, Any]]) -> str:
    """Render a human-readable list of tool names, descriptions, and schemas."""
    base_tools = [t for t in tools if isinstance(t, BaseTool)]
    if not base_tools:
        return "(no tools available)"
    return render_text_description_and_args(base_tools)


def render_tools_compact(tools: list[BaseTool | dict[str, Any]], max_tools: int | None = None) -> str:
    """Render a compact one-line-per-tool summary for small context window models.

    Instead of emitting the full argument schema (which can be hundreds of
    tokens per tool), this renders only the tool name and the first sentence
    of its description.  This is critical for on-device models like Apple
    Foundation Models whose total context window is ~4 096 tokens — at that
    size the full schema dump for 50+ tools exceeds the budget before any
    conversation message is included.

    Args:
        tools: The list of tools to render.
        max_tools: If set, only the first *max_tools* tools are included.
            Tools beyond this limit are silently dropped.  Pass ``None``
            to include all tools (still compact, but no hard cap).

    Returns:
        A compact bullet list, one tool per line.
    """
    base_tools = [t for t in tools if isinstance(t, BaseTool)]
    if not base_tools:
        return "(no tools available)"
    if max_tools is not None:
        base_tools = base_tools[:max_tools]
    lines: list[str] = []
    for tool in base_tools:
        desc = (tool.description or "").strip()
        # Keep only the first sentence to stay terse.
        first_sentence = desc.split(".")[0].strip()
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "…"
        lines.append(f"- {tool.name}: {first_sentence}")
    return "\n".join(lines)


# Tools that the ReAct planning guidance references by name.  When the tool
# list is capped for a small-context model these must stay visible, otherwise
# the system prompt tells the model to call a tool that was never rendered.
_PLANNING_TOOL_NAMES = ("write_todos",)


def prioritise_planning_tools(
    tools: list[BaseTool | dict[str, Any]],
) -> list[BaseTool | dict[str, Any]]:
    """Return *tools* reordered so planning tools (``write_todos``) come first.

    Order is otherwise preserved.  Used before a ``max_tools`` cap so the
    planning tool the ReAct prompt instructs the model to call is never the
    one that gets dropped.
    """
    def _name(t: BaseTool | dict[str, Any]) -> str:
        return getattr(t, "name", "") if not isinstance(t, dict) else t.get("name", "")

    planning = [t for t in tools if _name(t) in _PLANNING_TOOL_NAMES]
    if not planning:
        return list(tools)
    rest = [t for t in tools if _name(t) not in _PLANNING_TOOL_NAMES]
    return planning + rest


def normalise_args(
    action_input: str | dict[str, Any],
    tool: BaseTool | None,
) -> dict[str, Any]:
    """Coerce *action_input* to the ``dict[str, Any]`` form expected by ToolNode.

    When the model produces a plain string (e.g. just the query text) for a
    single-parameter tool, this function looks up the tool's schema to find the
    correct parameter name and wraps the string in ``{param_name: value}``.

    Args:
        action_input: The value from the parsed ``"action_input"`` JSON field.
        tool: The resolved ``BaseTool`` instance (may be ``None`` if the tool
              name was not recognised).

    Returns:
        A ``dict`` suitable for passing as ``tool_calls[].args``.
    """
    if isinstance(action_input, dict):
        return action_input

    # action_input is a plain string — infer the first parameter name
    if tool is not None:
        # Pydantic v2 path
        try:
            first_field = next(iter(tool.get_input_schema().model_fields))
            return {first_field: action_input}
        except (AttributeError, StopIteration):
            pass
        # Pydantic v1 / JSON Schema path
        try:
            props = tool.args_schema.schema().get("properties", {})
            first_field = next(iter(props))
            return {first_field: action_input}
        except (AttributeError, StopIteration):
            pass

    # Last resort: use a generic key
    return {"input": action_input}


def _extract_json_object(text: str) -> str | None:
    """Extract the first complete ``{...}`` from *text* using bracket counting.

    Correctly handles nested objects, strings, and escape sequences.  Unlike a
    regex, this never mis-identifies a nested closing brace as the end of the
    outermost object.

    Returns the raw JSON string (not parsed), or ``None`` if no complete object
    is found.
    """
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[: i + 1]
    return None


# Top-level keys in the action JSON that name the call rather than its args.
# Anything NOT in this set is treated as a top-level argument when the model
# emits the "flat args" shape ``{"action": "X", "query": "...", ...}`` instead
# of the canonical ``{"action": "X", "action_input": {"query": "..."}}``.
_ACTION_META_KEYS = frozenset({"action", "action_input", "input", "name"})


def _extract_action_input(blob: dict[str, Any]) -> str | dict[str, Any]:
    """Return the action arguments from a parsed action JSON blob.

    Accepts three shapes the model may emit:

    1. Canonical: ``{"action": "X", "action_input": {...}}``
    2. Legacy:    ``{"action": "X", "input": "..."}``
    3. Flat args: ``{"action": "X", "query": "...", "max_results": 5}``

    The flat-args shape is what small models (Llama, Qwen 4B, etc.) tend to
    produce because most of their training data uses OpenAI-function-calling
    flat arguments rather than LangChain's nested ``action_input`` wrapper.
    Without this fallback the parser silently dropped the arguments and the
    tool received an empty dict, triggering infinite retry loops in the
    agent (observed in production: see ``parse_action`` history).
    """
    if "action_input" in blob:
        return blob["action_input"]
    if "input" in blob:
        return blob["input"]
    flat_args = {k: v for k, v in blob.items() if k not in _ACTION_META_KEYS}
    if flat_args:
        return flat_args
    return ""


def parse_action(text: str) -> tuple[str, str | dict[str, Any]] | None:
    """Parse a ReAct *Action* block from *text*.

    Resolution order:
    1. Fenced ````` ```json ... ``` ````` format (preferred, model well-behaved).
    2. Inline JSON after ``Action:`` without a code fence.
    3. Bracket-counting fallback — finds the last ``Action:`` and extracts the
       first complete JSON object after it.  Handles missing closing fences,
       deeply nested args, and trailing text without being fooled by nested
       braces.

    The arguments are extracted by :func:`_extract_action_input`, which accepts
    the canonical ``action_input`` wrapper, the legacy ``input`` field, and
    the flat-args shape that small models often emit.

    Strips ``<think>`` blocks before searching.

    Returns:
        ``(tool_name, action_input)`` if an action block is found, else ``None``.
    """
    text = strip_think_tags(text)

    # ── Fast path: regex-based extraction ────────────────────────────────────
    for pattern in (_ACTION_FENCED_RE, _ACTION_INLINE_RE):
        match = pattern.search(text)
        if match:
            try:
                blob: dict[str, Any] = json.loads(match.group(1), strict=False)
            except json.JSONDecodeError:
                continue

            tool_name: str | None = blob.get("action") or blob.get("name")
            if tool_name:
                return tool_name, _extract_action_input(blob)

    # ── Robust fallback: bracket-counting after last "Action:" ────────────────
    # rfind ensures we pick up the *current* turn's action, not a historical
    # one rewritten into the conversation context.
    action_pos = text.lower().rfind("action:")
    if action_pos != -1:
        rest = text[action_pos + len("action:"):]
        brace_start = rest.find("{")
        if brace_start != -1:
            json_str = _extract_json_object(rest[brace_start:])
            if json_str:
                try:
                    blob = json.loads(json_str, strict=False)
                except json.JSONDecodeError:
                    pass
                else:
                    tool_name = blob.get("action") or blob.get("name")
                    if tool_name:
                        return tool_name, _extract_action_input(blob)

    return None


def reformat_tool_history(messages: list[AnyMessage]) -> list[BaseMessage]:
    """Convert structured tool-call history into ReAct-format message pairs.

    The deepagents framework stores completed tool interactions as::

        AIMessage(content="reasoning", tool_calls=[{name, args, id}])
        ToolMessage(content="result", tool_call_id=id)

    Text-only models cannot interpret ``tool_calls`` objects. This function
    rewrites those pairs into::

        AIMessage(content="reasoning\\nAction:\\n```json\\n{...}\\n```")
        HumanMessage(content="Observation: result")

    which is the standard ReAct conversational format understood by MLX models.
    Non-tool messages (``HumanMessage``, plain ``AIMessage``) are left unchanged.

    Args:
        messages: The full message list from the conversation.

    Returns:
        A new list with all tool-call pairs rewritten.
    """
    reformatted: list[BaseMessage] = []
    i = 0

    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Reconstruct the assistant's Thought + Action text
            thought_text: str = content_to_text(msg.content)
            action_lines: list[str] = []

            for tc in msg.tool_calls:
                action_blob = json.dumps(
                    {"action": tc["name"], "action_input": tc["args"]},
                    ensure_ascii=False,
                )
                action_lines.append(f"Action:\n```json\n{action_blob}\n```")

            full_text = thought_text
            if action_lines:
                full_text = (thought_text + "\n" + "\n".join(action_lines)).strip()

            reformatted.append(AIMessage(content=full_text))
            i += 1

            # Absorb the immediately following ToolMessages as Observations
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                obs = content_to_text(messages[i].content) if messages[i].content else ""
                reformatted.append(HumanMessage(content=f"Observation: {obs}"))
                i += 1

        else:
            reformatted.append(msg)
            i += 1

    return reformatted


def make_tool_call_id() -> str:
    """Generate a short unique ID for a synthesised tool call."""
    return f"react-{uuid.uuid4().hex[:8]}"
