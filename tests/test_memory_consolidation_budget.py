"""Memory consolidation must keep every LLM request within the model's budget.

Regression for a production incident: the consolidation pipeline
concatenated five session transcripts (one ~194 KB web-browsing session)
into ONE 153 020-token prompt for the local 9B MLX model — 11+ minutes of
prefill under the process-wide MLX lock, while every user chat waited.

Transcripts are written with the real ``append_event`` writer, so the
on-disk JSONL (``ensure_ascii`` escapes, full tool output, ``meta``) is
exactly what production reads.  The LLM is a fake that records requests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from backend import memory
from backend.config import AppConfig
from backend.session_transcript import append_event, list_transcripts_since

# Acceptance budget for one request to a local (MLX / oMLX / exo) model.
LOCAL_BUDGET_TOKENS = 16_384
# Tokens the consolidation call reserves for its JSON answer.
OUTPUT_RESERVE_TOKENS = 8192
# Conservative estimate, as in SmallContextTruncationMiddleware.  Measured
# with the Qwen3 tokenizer: decoded Russian ~3.6, English ~3.3 chars/token.
CHARS_PER_TOKEN = 3

RU = "Запомни: я предпочитаю утренние поезда и места у окна. "
EN = "Found three morning trains under budget; the 07:40 one is cheapest. "
PAGE_CHUNK = "<div class='row'>lorem ipsum dolor sit amet</div> "


def _tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)


class _FakeModel:
    """Records each request and answers like a well-behaved consolidator.

    Every call creates ``batch-<n>.md`` and appends ``- batch-<n>`` to the
    index it was shown, so tests can see whether batches fold together.
    """

    def __init__(self, profile: dict | None = None) -> None:
        self.profile = profile
        self.calls: list[str] = []

    async def ainvoke(self, messages):
        system, human = (m.content for m in messages)
        self.calls.append(system + human)
        n = len(self.calls)
        seen = system.split("<memory_index>\n", 1)[1].split("\n</memory_index>", 1)[0]
        index = "" if seen == "(empty)" else seen + "\n"
        doc = {
            "updates": [{
                "file": f"batch-{n}.md",
                "action": "create",
                "content": (
                    f"---\nname: batch-{n}\ndescription: facts from batch {n}\n"
                    f"type: project\n---\nBatch {n} facts.\n"
                ),
                "source_sessions": [],
                "confidence": "high",
                "reason": "test",
            }],
            "index_update": f"{index}- batch-{n}",
            "summary": f"batch {n}",
        }
        return AIMessage(content=json.dumps(doc))


def _write_transcript(sid: str, *, turns: int = 6) -> None:
    """~6.5k on-disk chars per turn: Russian request, browse, page dump, answer."""
    for t in range(turns):
        append_event(sid, "user", f"[{sid}-user-{t}] " + RU * 11, role="user")
        append_event(
            sid, "tool_call", {"url": f"https://example.com/{sid}/{t}"},
            tool_name="browser_navigate", tool_call_id=f"call_{t}",
        )
        append_event(
            sid, "tool_result", f"PAGE-DUMP-{sid}-{t} " + PAGE_CHUNK * 45,
            tool_name="browser_navigate", tool_call_id=f"call_{t}",
        )
        append_event(
            sid, "assistant", f"[{sid}-answer-{t}] " + EN * 13, role="assistant",
        )


@pytest.fixture
def consolidate(tmp_path, monkeypatch):
    """Run ``execute_consolidation`` over every transcript in a temp app dir."""
    for module in (
        "backend.config", "backend.memory",
        "backend.session_transcript", "backend.consolidation_lock",
    ):
        monkeypatch.setattr(f"{module}.get_app_data_dir", lambda: tmp_path)
    monkeypatch.setattr(memory, "_status", memory.MemoryStatus())
    # Module-level Event binds to the first loop that awaits it.
    monkeypatch.setattr(memory, "_cancel_event", asyncio.Event())
    monkeypatch.setattr(AppConfig, "apply_to_environ", lambda self: None)

    async def _run(
        *, provider: str = "mlx", profile: dict | None = None,
        model: _FakeModel | None = None,
    ):
        cfg = AppConfig()
        cfg.llm.provider = provider
        monkeypatch.setattr(AppConfig, "aload", AsyncMock(return_value=cfg))
        fake = model or _FakeModel(profile)
        monkeypatch.setattr(
            "backend.memory_relevance._create_ranking_model",
            lambda *args, **kwargs: fake,
        )
        await memory.execute_consolidation(
            0.0, list_transcripts_since(0), cfg.memory,
        )
        return fake

    return _run


SESSIONS = [f"sess{i}-0000-aaaa" for i in range(5)]


def _assert_all_turns_covered(calls: list[str], sessions: list[str], turns: int) -> None:
    sent = "\n".join(calls)
    for sid in sessions:
        assert f"--- Session {sid}" in sent, f"session {sid} never sent"
        for t in range(turns):
            assert f"[{sid}-user-{t}]" in sent, f"{sid} turn {t} request missing"
            assert f"[{sid}-answer-{t}]" in sent, f"{sid} turn {t} answer missing"


async def test_local_model_requests_stay_within_budget_and_cover_all_transcripts(consolidate):
    for sid in SESSIONS:
        _write_transcript(sid)

    fake = await consolidate(provider="mlx")

    assert fake.calls, "consolidation never called the model"
    sizes = [_tokens(c) for c in fake.calls]
    assert max(sizes) <= LOCAL_BUDGET_TOKENS, f"request sizes (tokens): {sizes}"
    _assert_all_turns_covered(fake.calls, SESSIONS, turns=6)
    status = memory.get_status()
    assert status.state == memory.RunState.SUCCESS, status.error
    assert status.transcripts_processed == len(SESSIONS)


async def test_later_batches_see_the_index_and_topics_written_by_earlier_ones(consolidate, tmp_path):
    for sid in SESSIONS:
        _write_transcript(sid)

    fake = await consolidate(provider="mlx")

    assert len(fake.calls) >= 2, "expected the transcripts to be split into batches"
    for n in range(2, len(fake.calls) + 1):
        request = fake.calls[n - 1]
        assert f"- batch-{n - 1}" in request.split("</memory_index>")[0]
        assert f"batch-{n - 1}: facts from batch {n - 1}" in request.split("</topic_files>")[0]
    index = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert index.splitlines() == [f"- batch-{n}" for n in range(1, len(fake.calls) + 1)]


class _HangingModel(_FakeModel):
    """A generation that only ends when the run is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def ainvoke(self, messages):
        self.calls.append(messages[0].content)
        self.started.set()
        await asyncio.Event().wait()


async def test_cancel_during_a_request_stops_the_run(consolidate):
    _write_transcript(SESSIONS[0], turns=1)
    model = _HangingModel()

    async def cancel_once_started():
        await model.started.wait()
        memory.request_cancel()

    canceller = asyncio.create_task(cancel_once_started())
    await consolidate(model=model)
    await canceller

    assert len(model.calls) == 1
    assert memory.get_status().state == memory.RunState.CANCELLED


async def test_cancel_between_batches_starts_no_further_request(consolidate, monkeypatch):
    for sid in SESSIONS:
        _write_transcript(sid)
    loop = asyncio.get_running_loop()
    apply_updates = memory._apply_updates

    def apply_then_cancel(result):
        # The user presses Cancel while batch 1 is being written to disk.
        loop.call_soon_threadsafe(memory.request_cancel)
        return apply_updates(result)

    monkeypatch.setattr(memory, "_apply_updates", apply_then_cancel)
    fake = await consolidate(provider="mlx")

    assert len(fake.calls) == 1, "a cancelled run must not start the next batch"
    assert memory.get_status().state == memory.RunState.CANCELLED


async def test_single_oversized_transcript_is_clipped_head_and_tail_and_logged(consolidate, caplog):
    sid = "huge-session-0001"
    _write_transcript(sid, turns=40)  # ~24k tokens of requests and answers

    with caplog.at_level(logging.WARNING, logger="backend.memory"):
        fake = await consolidate(provider="mlx")

    assert len(fake.calls) == 1
    request = fake.calls[0]
    assert _tokens(request) <= LOCAL_BUDGET_TOKENS
    assert f"--- Session {sid}" in request
    assert f"[{sid}-user-0]" in request, "the opening request must survive clipping"
    assert f"[{sid}-answer-39]" in request, "the final answer must survive clipping"
    assert any(
        r.levelno == logging.WARNING and sid in r.getMessage() for r in caplog.records
    ), "clipping an oversized transcript must be logged"


async def test_budget_shrinks_to_the_models_advertised_input_window(consolidate):
    for sid in SESSIONS:
        _write_transcript(sid)

    window = 20_000
    fake = await consolidate(provider="mlx", profile={"max_input_tokens": window})

    sizes = [_tokens(c) for c in fake.calls]
    assert max(sizes) <= window - OUTPUT_RESERVE_TOKENS, f"request sizes (tokens): {sizes}"
    sent = "\n".join(fake.calls)
    assert all(f"--- Session {sid}" in sent for sid in SESSIONS)


async def test_cloud_model_keeps_a_larger_budget(consolidate):
    for sid in SESSIONS:
        _write_transcript(sid)

    fake = await consolidate(provider="anthropic")

    assert len(fake.calls) == 1, "all five sessions fit one cloud request"
    assert _tokens(fake.calls[0]) > LOCAL_BUDGET_TOKENS
    _assert_all_turns_covered(fake.calls, SESSIONS, turns=6)


async def test_oversized_memory_index_fails_fast_without_calling_the_model(consolidate, tmp_path):
    _write_transcript(SESSIONS[0], turns=1)
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "MEMORY.md").write_text("- fact\n" * 9000, encoding="utf-8")

    fake = await consolidate(provider="mlx")

    assert fake.calls == []
    status = memory.get_status()
    assert status.state == memory.RunState.ERROR
    assert "MEMORY.md" in (status.error or "")


async def test_transcript_is_compacted_to_its_informative_parts(consolidate):
    sid = "compact-session-01"
    append_event(sid, "user", "Запомни: мой любимый цвет — изумрудный.", role="user")
    append_event(
        sid, "tool_call", {"query": "emerald colour hex"},
        tool_name="web_search", tool_call_id="call_1",
    )
    append_event(
        sid, "tool_result", "PAGE-HEAD " + PAGE_CHUNK * 1000,
        tool_name="web_search", tool_call_id="call_1",
    )
    append_event(
        sid, "tool_result", "Captured the screen.", tool_name="read_screen",
        tool_call_id="call_2",
        metadata={"subagent": "vision", "images": [{"base64": "QUJD" * 25_000}]},
    )
    append_event(sid, "assistant", "Emerald is #50C878 — noted as your favourite.", role="assistant")

    fake = await consolidate(provider="mlx")

    request = fake.calls[0]
    assert "мой любимый цвет — изумрудный" in request, "Cyrillic must reach the model decoded"
    assert "\\u04" not in request, "no JSON unicode escapes"
    assert "Emerald is #50C878" in request
    assert "web_search" in request and "emerald colour hex" in request
    assert "read_screen" in request
    assert PAGE_CHUNK * 10 not in request, "bulky tool output must be dropped"
    assert "QUJDQUJD" not in request, "base64 image payloads must be dropped"
    assert len(request) < 10_000
