"""Native tool-call parsing for tool-aware HuggingFace chat templates.

Modern open-weight models (Qwen 2.5+/3, Llama 3.1+, Mistral-Nemo, Hermes-3,
Phi-3.5+) ship Jinja chat templates that accept a ``tools=`` slot and were
fine-tuned to emit family-specific structured tool-call markers in their
output.  This module:

* Inspects a tokenizer's chat template to decide whether the underlying model
  was trained for tool calling, and which family marker it uses.
* Parses raw decoded text emitted by such models into LangChain
  ``tool_calls`` dicts.

The detection is deliberately conservative — when in doubt we return
``False`` and the caller falls back to the text-based ReAct shim.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)


# ── Family detection ─────────────────────────────────────────────────────────

# Each family uses different scaffolding to express tool calls.  Detection
# scans the chat-template Jinja source for a marker unique to that family.
_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    # Order matters — Hermes/Qwen share a marker, so check Hermes first when
    # the template references it explicitly.  In practice both produce the
    # same ``<tool_call>{json}</tool_call>`` output, so this is mostly cosmetic.
    ("qwen", "<tool_call>"),
    ("mistral", "[TOOL_CALLS]"),
    ("llama", "<|python_tag|>"),
)


def detect_native_tool_support(tokenizer: Any) -> tuple[bool, str]:
    """Return ``(supported, family)`` for a tokenizer's chat template.

    A model is considered tool-aware when its template both references the
    ``tools`` Jinja variable AND contains a family-specific tool-call marker.
    The ``tools`` check alone is too permissive — some templates accept a
    ``tools`` argument purely for documentation and never emit a structured
    call marker, so the model would silently regress to text output.

    Returns ``(False, "unknown")`` for any template we don't recognise so the
    caller transparently falls back to the ReAct text shim.
    """
    template = getattr(tokenizer, "chat_template", None) or ""
    if not template:
        return False, "unknown"

    template_lower = template.lower()
    if "tools" not in template_lower:
        return False, "unknown"

    for family, marker in _FAMILY_MARKERS:
        if marker.lower() in template_lower:
            return True, family

    return False, "unknown"


# ── Stop tokens per family ───────────────────────────────────────────────────
#
# Returning these to the streaming loop lets us terminate generation as soon
# as the model finishes its turn, without waiting for the model to keep
# emitting tokens past the EOS marker.

_FAMILY_STOP_TOKENS: dict[str, tuple[str, ...]] = {
    # Qwen / Hermes — ChatML EOS markers
    "qwen": ("<|im_end|>",),
    # Llama 3 family — Llama 3.1+ uses <|eot_id|> for tool-call turn boundaries
    "llama": ("<|eot_id|>", "<|end_of_text|>"),
    # Mistral / Mixtral — closing instruction tag (<|/INST|> on Nemo)
    "mistral": ("</s>",),
    "unknown": (),
}


def stop_tokens_for(family: str) -> tuple[str, ...]:
    """Return the chat-template control tokens that mark turn-end for *family*."""
    return _FAMILY_STOP_TOKENS.get(family, ())


# ── Tool-call markers per family ─────────────────────────────────────────────
#
# Each parser locates the marker, then uses bracket-counting to extract a
# complete JSON payload — regex alone can't reliably handle nested braces or
# braces-in-strings inside tool arguments.

_QWEN_OPEN = re.compile(r"<tool_call>", re.IGNORECASE)
_LLAMA_OPEN = re.compile(r"<\|python_tag\|>", re.IGNORECASE)
_MISTRAL_OPEN = re.compile(r"\[TOOL_CALLS\]", re.IGNORECASE)

# Hermes / Functionary XML-style tool call (used by some Qwen3.5 fine-tunes
# such as ``mlx-community/Qwen3.5-4B-OptiQ-4bit``).  The wrapping
# ``<tool_call>...</tool_call>`` is the same as standard Qwen — only the
# *interior* changes from JSON to XML.  Example::
#
#     <tool_call>
#     <function=add>
#     <parameter=a>17</parameter>
#     <parameter=b>25</parameter>
#     </function>
#     </tool_call>
_QWEN_XML_BLOCK = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>([\s\S]*?)</function>\s*</tool_call>",
    re.IGNORECASE,
)
_XML_PARAM = re.compile(
    r"<parameter=([^>\s]+)>([\s\S]*?)</parameter>",
    re.IGNORECASE,
)


def _extract_balanced_json(text: str, start: int) -> tuple[str | None, int]:
    """Extract the first complete JSON value (object or array) starting at *start*.

    Returns ``(json_str, end_index)`` where ``end_index`` is the position
    immediately after the closing bracket, or ``(None, start)`` when no
    complete value is found.

    Correctly handles nested objects/arrays, strings, and escape sequences —
    a regex like ``\\{.*?\\}`` cannot, and would mis-cut tool calls whose
    arguments contain stringified JSON, code snippets, or nested dicts.
    """
    # Skip leading whitespace
    i = start
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    if i >= len(text):
        return None, start

    open_ch = text[i]
    if open_ch not in "{[":
        return None, start
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_string = False
    escape_next = False
    for j in range(i, len(text)):
        ch = text[j]
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
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[i:j + 1], j + 1
    return None, start


def _make_call_id() -> str:
    """Generate a short ``call_<8hex>`` id matching the OpenAI tool-call convention."""
    return f"call_{uuid.uuid4().hex[:8]}"


def _coerce_tool_call(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a parsed JSON tool-call blob into the LangChain ``tool_calls`` shape.

    Accepts the union of conventions used across families:

    * ``{"name": ..., "arguments": {...}}``    — Qwen, Hermes, OpenAI standard
    * ``{"name": ..., "parameters": {...}}``   — Llama 3
    * ``{"name": ..., "arguments": "json str"}`` — some Mistral variants

    Returns ``None`` when the blob lacks a ``name`` field.
    """
    name = raw.get("name") or raw.get("function") or raw.get("tool")
    if not name:
        return None

    args: Any = raw.get("arguments")
    if args is None:
        args = raw.get("parameters")
    if args is None:
        args = raw.get("args", {})

    # Some models stringify the arguments dict — accept either form
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"input": args}
    if not isinstance(args, dict):
        args = {"input": args}

    call_id = raw.get("id") or _make_call_id()
    return {"name": str(name), "args": args, "id": call_id, "type": "tool_call"}


def _parse_with_marker(text: str, marker_re: re.Pattern[str]) -> list[dict[str, Any]]:
    """Find every ``marker_re`` in *text* and parse the JSON payload after each.

    Used by Qwen, Llama and Mistral parsers — they only differ in the marker.
    Multiple tool calls in one turn (parallel calls) are handled naturally.
    """
    calls: list[dict[str, Any]] = []
    for match in marker_re.finditer(text):
        json_str, _ = _extract_balanced_json(text, match.end())
        if not json_str:
            continue
        try:
            blob = json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            logger.debug("native tool-call JSON decode failed: %s", e)
            continue

        # Some families emit a single object, others wrap calls in an array
        items = blob if isinstance(blob, list) else [blob]
        for item in items:
            if not isinstance(item, dict):
                continue
            tc = _coerce_tool_call(item)
            if tc is not None:
                calls.append(tc)
    return calls


def _parse_qwen_xml(text: str) -> list[dict[str, Any]]:
    """Parse Hermes/Functionary XML-style ``<tool_call>...</tool_call>`` blocks.

    Some Qwen3.5 fine-tunes (e.g. ``Qwen3.5-4B-OptiQ-4bit``) replace the
    standard JSON payload with nested ``<function>``/``<parameter>`` tags.
    Argument values are returned as strings; :func:`_coerce_tool_call`'s
    JSON-loads pass will recover ints/floats/bools when the value parses as JSON.
    """
    calls: list[dict[str, Any]] = []
    for match in _QWEN_XML_BLOCK.finditer(text):
        name = match.group(1).strip()
        body = match.group(2)
        args: dict[str, Any] = {}
        for pm in _XML_PARAM.finditer(body):
            key = pm.group(1).strip()
            raw_val = pm.group(2).strip()
            # Try to recover non-string types when the value parses as JSON
            try:
                args[key] = json.loads(raw_val)
            except (json.JSONDecodeError, ValueError):
                args[key] = raw_val
        calls.append({
            "name": name, "args": args, "id": _make_call_id(), "type": "tool_call",
        })
    return calls


def parse_native_tool_calls(text: str, family: str) -> list[dict[str, Any]] | None:
    """Extract structured tool calls from *text* using the *family* parser.

    Returns ``None`` when *family* is unknown (no parsing attempted) or when
    the text contained no tool-call markers — this lets the caller distinguish
    "no calls in this turn" (return ``None``, treat as final answer) from
    "calls present" (return list, populate ``AIMessage.tool_calls``).

    For the ``qwen`` family both the standard JSON payload and the
    Hermes/Functionary XML payload are supported — small fine-tuned variants
    use the latter.

    The text is searched in full; ``<think>`` blocks are NOT stripped here
    because the caller decides whether thinking-mode tool drafts should be
    executed.  Most callers should call :func:`strip_think_tags` before
    passing the text in.
    """
    if family == "qwen":
        # Try the XML form first — it's unambiguous.  When it doesn't match,
        # fall back to the standard JSON form.
        calls = _parse_qwen_xml(text)
        if not calls:
            calls = _parse_with_marker(text, _QWEN_OPEN)
    elif family == "llama":
        calls = _parse_with_marker(text, _LLAMA_OPEN)
    elif family == "mistral":
        calls = _parse_with_marker(text, _MISTRAL_OPEN)
    else:
        return None

    return calls or None


# ── Cleanup ──────────────────────────────────────────────────────────────────


def strip_tool_call_markup(text: str, family: str) -> str:
    """Remove tool-call markup from *text* so the user-visible content is clean.

    The model may emit content alongside (or interleaved with) tool calls.
    For display we want the surrounding prose without the JSON scaffold.
    """
    if family == "qwen":
        return re.sub(
            r"<tool_call>[\s\S]*?</tool_call>", "", text, flags=re.IGNORECASE
        ).strip()
    if family == "llama":
        # Llama 3 tool calls run from <|python_tag|> to <|eom_id|> (or EOS)
        return re.sub(
            r"<\|python_tag\|>[\s\S]*?(?:<\|eom_id\|>|$)", "", text, flags=re.IGNORECASE
        ).strip()
    if family == "mistral":
        return re.sub(
            r"\[TOOL_CALLS\][\s\S]*?(?:</s>|$)", "", text, flags=re.IGNORECASE
        ).strip()
    return text
