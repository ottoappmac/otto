"""Live run-timeline folding: agent_delta tokens must not become empty rows.

Mirrors ``app/src/utils/liveTimeline.ts``. The session run page used to
append every streamed token as its own timeline event, which rendered as
a rail of empty bullets. Deltas accumulate into one assistant event; the
authoritative ``agent`` frame replaces that placeholder; a following
tool call keeps any preamble as a completed thought.
"""

from __future__ import annotations


SKIP_TYPES = {
    "execute_output",
    "memory_search",
    "memory_context",
    "context_received",
}


def _is_streaming_assistant(ev: dict | None) -> bool:
    return bool(ev and ev.get("type") == "assistant" and (ev.get("meta") or {}).get("streaming"))


def _to_event(raw: dict, extra: dict | None = None) -> dict:
    meta = raw.get("metadata") or {}
    ev = {
        "type": "assistant" if raw.get("type") in ("agent", "agent_delta") else raw.get("type"),
        "content": raw.get("content"),
        "subagent": meta.get("subagent"),
        "args": meta.get("args"),
        "tool": meta.get("name") if raw.get("type") == "tool_result" else (
            raw.get("content") if isinstance(raw.get("content"), str) else None
        ),
        "tool_call_id": meta.get("tool_call_id"),
        "images": meta.get("images"),
    }
    if extra:
        ev.update(extra)
    return {k: v for k, v in ev.items() if v is not None}


def _finalize_streaming(prev: list[dict]) -> list[dict]:
    if not prev:
        return prev
    last = prev[-1]
    if not _is_streaming_assistant(last):
        return prev
    content = last.get("content") or ""
    if not str(content).strip():
        return prev[:-1]
    updated = list(prev)
    meta = dict(last.get("meta") or {})
    meta["streaming"] = False
    updated[-1] = {**last, "meta": meta}
    return updated


def _apply(prev: list[dict], raw: dict) -> list[dict]:
    if raw.get("type") == "agent_delta":
        piece = raw.get("content") if isinstance(raw.get("content"), str) else ""
        if not piece:
            return prev
        last = prev[-1] if prev else None
        if _is_streaming_assistant(last):
            updated = list(prev)
            updated[-1] = {**last, "content": str(last.get("content") or "") + piece}
            return updated
        return [*prev, _to_event(raw, {"content": piece, "meta": {"streaming": True}})]

    if raw.get("type") in SKIP_TYPES:
        return prev

    if raw.get("type") == "agent":
        last = prev[-1] if prev else None
        if _is_streaming_assistant(last):
            updated = list(prev)
            updated[-1] = _to_event(raw)
            return updated

    return [*_finalize_streaming(prev), _to_event(raw)]


def test_deltas_accumulate_into_one_assistant_row():
    events: list[dict] = []
    for piece in ("Here", "'s the ", "brutal truth"):
        events = _apply(events, {"type": "agent_delta", "content": piece})
    assert len(events) == 1
    assert events[0]["type"] == "assistant"
    assert events[0]["content"] == "Here's the brutal truth"
    assert events[0]["meta"]["streaming"] is True


def test_empty_delta_is_ignored():
    events = _apply([], {"type": "agent_delta", "content": ""})
    assert events == []


def test_agent_frame_replaces_streaming_placeholder():
    events = _apply([], {"type": "agent_delta", "content": "partial"})
    events = _apply(events, {"type": "agent", "content": "Here's the brutal truth after stress-testing."})
    assert len(events) == 1
    assert events[0]["content"] == "Here's the brutal truth after stress-testing."
    assert not (events[0].get("meta") or {}).get("streaming")


def test_tool_call_keeps_streamed_thought():
    events = _apply([], {"type": "agent_delta", "content": "I'll inspect the packages first."})
    events = _apply(events, {"type": "tool_call", "content": "execute"})
    assert [e["type"] for e in events] == ["assistant", "tool_call"]
    assert events[0]["content"] == "I'll inspect the packages first."
    assert events[0]["meta"]["streaming"] is False


def test_empty_placeholder_is_dropped_on_tool_call():
    events = _apply([], {"type": "agent_delta", "content": "  "})
    events = _apply(events, {"type": "tool_call", "content": "execute"})
    assert [e["type"] for e in events] == ["tool_call"]


def test_noise_types_are_skipped():
    events: list[dict] = []
    for t in ("execute_output", "memory_search", "memory_context", "context_received"):
        events = _apply(events, {"type": t, "content": "noise"})
    assert events == []
