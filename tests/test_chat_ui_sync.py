"""Chat UI transcript helpers: thoughts stay, HITL cards are not dropped.

Mirrors ``app/src/utils/mergeToolMessages.ts`` so the two bugs stay covered
without a frontend test runner: (1) intermediate agent text is still a
thought after tools/HITL, (2) a persisted interrupt is appended even when
local state is longer than the API transcript.
"""

from __future__ import annotations


def _pending_interrupts(messages: list[dict]) -> list[dict]:
    return [
        m for m in messages
        if m.get("type") in ("hitl_request", "ask_user") and not (m.get("metadata") or {}).get("resolved")
    ]


def _append_missing_interrupts(prev: list[dict], api_merged: list[dict]) -> list[dict] | None:
    api_hitl = _pending_interrupts(api_merged)
    if not api_hitl:
        return None
    if _pending_interrupts(prev):
        return None
    return [*prev, *api_hitl]


def _thought_flags(messages: list[dict]) -> list[bool]:
    flags = [False] * len(messages)
    has_follow_up = False
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.get("type") == "agent" and not (m.get("metadata") or {}).get("subagent"):
            flags[i] = has_follow_up
            has_follow_up = True
        elif m.get("type") in ("tool_call", "tool_result", "hitl_request", "ask_user"):
            has_follow_up = True
        elif m.get("type") == "user" and not (m.get("metadata") or {}).get("isContext"):
            has_follow_up = False
    return flags


def _finalize_streaming_thought(prev: list[dict]) -> list[dict]:
    if not prev:
        return prev
    last = prev[-1]
    if last.get("type") != "agent" or not (last.get("metadata") or {}).get("streaming"):
        return prev
    if not str(last.get("content") or "").strip():
        return prev[:-1]
    updated = list(prev)
    meta = dict(last.get("metadata") or {})
    meta["streaming"] = False
    updated[-1] = {**last, "metadata": meta}
    return updated


def test_tool_call_keeps_streamed_thought():
    prev = [
        {"type": "user", "content": "run pip list"},
        {"type": "agent", "content": "I'll inspect the packages first.", "metadata": {"streaming": True}},
    ]
    kept = _finalize_streaming_thought(prev)
    kept.append({"type": "tool_call", "content": "execute"})
    assert kept[1]["content"] == "I'll inspect the packages first."
    assert kept[1]["metadata"]["streaming"] is False
    flags = _thought_flags(kept)
    assert flags == [False, True, False]


def test_empty_streaming_placeholder_is_dropped():
    prev = [
        {"type": "user", "content": "hi"},
        {"type": "agent", "content": "  ", "metadata": {"streaming": True}},
    ]
    assert _finalize_streaming_thought(prev) == [{"type": "user", "content": "hi"}]


def test_poll_appends_hitl_when_local_transcript_is_longer():
    prev = [
        {"type": "user", "content": "run it"},
        {"type": "agent", "content": "Running the command."},
        {"type": "tool_call", "content": "execute"},
    ]
    api = [
        {"type": "user", "content": "run it"},
        {"type": "tool_call", "content": "execute"},
        {
            "type": "hitl_request",
            "content": "Tool execution requires approval",
            "metadata": {"action_requests": [{"name": "execute"}]},
        },
    ]
    merged = _append_missing_interrupts(prev, api)
    assert merged is not None
    assert merged[-1]["type"] == "hitl_request"
    assert merged[1]["content"] == "Running the command."
    flags = _thought_flags(merged)
    assert flags[1] is True


def test_poll_does_not_duplicate_existing_hitl():
    hitl = {"type": "hitl_request", "content": "Tool execution requires approval", "metadata": {}}
    prev = [{"type": "tool_call", "content": "execute"}, hitl]
    api = [{"type": "tool_call", "content": "execute"}, hitl]
    assert _append_missing_interrupts(prev, api) is None


def test_thought_flags_treat_tool_result_and_hitl_as_follow_up():
    messages = [
        {"type": "agent", "content": "Checking."},
        {"type": "tool_result", "content": "execute"},
        {"type": "hitl_request", "content": "Tool execution requires approval"},
    ]
    assert _thought_flags(messages) == [True, False, False]


def _group_consecutive_thoughts(flags: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = -1
    for i, flag in enumerate(flags):
        if flag:
            if start == -1:
                start = i
        elif start != -1:
            runs.append((start, i - 1))
            start = -1
    if start != -1:
        runs.append((start, len(flags) - 1))
    return runs


def test_consecutive_thoughts_collapse_into_one_run():
    messages = [
        {"type": "user", "content": "review this"},
        {"type": "agent", "content": "I'll read the pack first."},
        {"type": "agent", "content": "Then I'll cross-check the rates."},
        {"type": "tool_call", "content": "doc_reader"},
        {"type": "agent", "content": "Now the questionnaire."},
        {"type": "tool_call", "content": "doc_reader"},
        {"type": "agent", "content": "Here is the review."},
    ]
    flags = _thought_flags(messages)
    assert flags == [False, True, True, False, True, False, False]
    assert _group_consecutive_thoughts(flags) == [(1, 2), (4, 4)]


def test_prior_turn_answer_stays_visible_after_new_user_message():
    """The last streamed agent text of a turn must not be demoted to a
    collapsed thought when the user continues the session."""
    messages = [
        {"type": "user", "content": "review the tender"},
        {"type": "agent", "content": "I'll read the documents first."},
        {"type": "tool_call", "content": "doc_reader"},
        {"type": "agent", "content": "The pack is well-structured. Here's the review."},
        {"type": "user", "content": "what about the savings figure?"},
        {"type": "agent", "content": "I'll check the comparison table."},
        {"type": "tool_call", "content": "read_file"},
        {"type": "agent", "content": "The $17k figure is gross, not net."},
    ]
    flags = _thought_flags(messages)
    # 0 user, 1 thought, 2 tool, 3 ANSWER, 4 user, 5 thought, 6 tool, 7 ANSWER
    assert flags[1] is True
    assert flags[3] is False
    assert flags[5] is True
    assert flags[7] is False


def test_context_injection_does_not_reset_thought_turn():
    messages = [
        {"type": "user", "content": "run it"},
        {"type": "agent", "content": "I'll execute that."},
        {"type": "user", "content": "also check the logs", "metadata": {"isContext": True}},
        {"type": "tool_call", "content": "execute"},
    ]
    flags = _thought_flags(messages)
    assert flags[1] is True
