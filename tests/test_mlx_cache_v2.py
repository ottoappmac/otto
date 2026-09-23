"""Prefix-cache v2: per-agent snapshot families and a reuse point at the turn start.

Two failures seen in the real app (Qwen3.8-9B, a qwen3_5_text hybrid):

* The orchestrator and a browser agent running at the same time each stored
  their two static snapshots (tool block + system turn) in a process-wide LRU
  of two entries, so each evicted the other's and every orchestrator turn
  paid ~64k tokens of cold prefill.  The store now keeps a family of entries
  per bound tool set and evicts whole families, least recently used first.
* A follow-up (or an agent step whose observation arrives as a user message)
  re-renders the previous assistant turn: the Qwen3.5 template drops the
  empty ``<think>\\n\\n</think>\\n\\n`` block from assistant turns before the
  last user query.  The new prompt diverges 3 tokens after the previous
  generation prompt started — before the ``len(prompt) - 1`` boundary 18bb476
  snapshots — so a hybrid model rebuilt its cache.  Prompts that end with a
  user turn now also get a reuse point where the generation prompt starts.

Built on the harness of ``test_mlx_prefix_cache`` / ``test_mlx_static_prefix``:
the real ``mlx_lm`` generation loop on a tiny deterministic model with real
``ArraysCache`` / ``KVCache`` layers.  Every step asserts that the cache
handed to the model equals a fresh prefill of the tokens it claims to hold.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from chat_models.mlx import _prefix_disk, _prefix_store, chat_mlx_text  # noqa: E402
from chat_models.mlx.chat_mlx_text import ChatMLXText  # noqa: E402
from tests.test_mlx_prefix_cache import ALL_KINDS, HYBRID  # noqa: E402
from tests.test_mlx_static_prefix import (  # noqa: E402
    _reused,
    _run,
    _Sessions,
    _static_ends,
    _tools,
    _turn_start,
)


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """A fresh process-wide RAM store for every test, and no SSD tier."""
    fresh = _prefix_store.PrefixSnapshotStore()
    monkeypatch.setattr(_prefix_store, "STATIC_PREFIXES", fresh)
    monkeypatch.setattr(_prefix_disk, "SSD_PREFIXES", _prefix_disk.DiskPrefixStore(None))
    return fresh


def _diverged(h, before, after, tools):
    a = h.tokenizer.encode(h.llm._to_prompt(before, tools=tools))
    b = h.tokenizer.encode(h.llm._to_prompt(after, tools=tools))
    return next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)


def _source(ai):
    return ai.response_metadata["prefix_cache_source"]


# ── Store: one family of snapshots per bound tool set ─────────────────────────


class _Model:
    """Any object that can be weakly referenced stands in for a loaded model."""


def test_second_family_never_evicts_the_first_familys_pair():
    store = _prefix_store.PrefixSnapshotStore(max_entries=2, max_families=4)
    model = _Model()
    store.put(model, "kv4", [1, 2], ["orchestrator tools"], family="orchestrator")
    store.put(model, "kv4", [1, 2, 3], ["orchestrator system"], family="orchestrator")
    store.put(model, "kv4", [7, 8], ["browser tools"], family="browser")
    store.put(model, "kv4", [7, 8, 9], ["browser system"], family="browser")
    assert len(store) == 4
    assert store.get(model, "kv4", [1, 2]) == ["orchestrator tools"]
    assert store.get(model, "kv4", [1, 2, 3]) == ["orchestrator system"]
    assert store.get(model, "kv4", [7, 8]) == ["browser tools"]
    assert store.get(model, "kv4", [7, 8, 9]) == ["browser system"]


def test_a_familys_third_entry_evicts_only_within_that_family():
    store = _prefix_store.PrefixSnapshotStore(max_entries=2, max_families=4)
    model = _Model()
    store.put(model, "kv4", [1], ["a1"], family="a")
    store.put(model, "kv4", [2], ["a2"], family="a")
    store.put(model, "kv4", [7], ["b1"], family="b")
    store.put(model, "kv4", [8], ["b2"], family="b")
    store.put(model, "kv4", [3], ["a3"], family="a")
    assert store.get(model, "kv4", [1]) is None
    assert [store.get(model, "kv4", [t]) for t in (2, 3, 7, 8)] == [["a2"], ["a3"], ["b1"], ["b2"]]


def test_least_recently_used_family_is_evicted_first():
    store = _prefix_store.PrefixSnapshotStore(max_entries=2, max_families=2)
    model = _Model()
    store.put(model, "kv4", [1], ["a"], family="a")
    store.put(model, "kv4", [2], ["b"], family="b")
    assert store.get(model, "kv4", [1]) == ["a"]  # family a is now the most recently used
    store.put(model, "kv4", [3], ["c"], family="c")
    assert store.get(model, "kv4", [2]) is None
    assert store.get(model, "kv4", [1]) == ["a"]
    assert store.get(model, "kv4", [3]) == ["c"]


def test_memory_cap_evicts_least_recently_used_families_but_keeps_the_newest():
    store = _prefix_store.PrefixSnapshotStore(max_entries=2, max_families=4, max_bytes=100)
    model = _Model()
    store.put(model, "kv4", [1], ["a"], family="a", nbytes=60)
    store.put(model, "kv4", [2], ["b"], family="b", nbytes=60)
    assert store.get(model, "kv4", [1]) is None
    assert store.get(model, "kv4", [2]) == ["b"]
    # A single family over the cap is still kept: it is the one in use.
    store.put(model, "kv4", [3], ["b2"], family="b", nbytes=60)
    assert store.get(model, "kv4", [2]) == ["b"]
    assert store.get(model, "kv4", [3]) == ["b2"]


_MB = 1024**2


def test_the_ram_store_keeps_two_agents_within_1_5_gib_by_default():
    # Idle snapshots stay resident next to the weights, and the SSD tier
    # restores an evicted family's tool block in ~0.1-0.3 s.
    store = _prefix_store.PrefixSnapshotStore()
    model = _Model()
    # 4-bit Qwen3.8-9B: the orchestrator's and the browser agent's tool block + system turn.
    store.put(model, "kv4", [1], ["o tools"], family="orchestrator", nbytes=356 * _MB)
    store.put(model, "kv4", [1, 2], ["o system"], family="orchestrator", nbytes=370 * _MB)
    store.put(model, "kv4", [7], ["b tools"], family="browser", nbytes=286 * _MB)
    store.put(model, "kv4", [7, 8], ["b system"], family="browser", nbytes=318 * _MB)
    assert len(store) == 4
    # A third agent evicts the least recently used one.
    store.put(model, "kv4", [5], ["c tools"], family="coder", nbytes=10 * _MB)
    assert store.family_count == 2
    assert store.get(model, "kv4", [1]) is None
    # A bf16 family (~1.1 GB per entry) is kept alone.
    store.put(model, "kv4", [9], ["bf16 tools"], family="bf16", nbytes=1100 * _MB)
    store.put(model, "kv4", [9, 10], ["bf16 system"], family="bf16", nbytes=1100 * _MB)
    assert store.family_count == 1


def _browser_run(question="Open example.com and read the title."):
    return [
        SystemMessage("You are the browser agent. Report what the page shows. " * 3),
        HumanMessage(question),
    ]


_BROWSER_TOOLS = ("navigate", "click_element", "type_text", "screenshot", "scroll_page")


@pytest.mark.parametrize("kinds", [HYBRID], ids=["hybrid"])
def test_concurrent_agent_families_keep_their_own_static_prefixes(monkeypatch, kinds):
    # The orchestrator and a browser agent alternate on one loaded model.
    h = _Sessions(monkeypatch, kinds)
    orchestrator_tools, browser_tools = _tools(), _tools(*_BROWSER_TOOLS)
    h.step(next(_run(session="s1")), orchestrator_tools)
    h.new_session()
    h.step(_browser_run(), browser_tools)
    for round_ in range(2):
        h.new_session()
        messages = next(_run(session="s1", question=f"Anything new ({round_})?"))
        _, _, ai = h.step(messages, orchestrator_tools)
        assert _reused(ai) == _static_ends(h, messages, orchestrator_tools)[1]
        assert _source(ai) == "ram"
        h.new_session()
        messages = _browser_run(f"Open example.org ({round_}).")
        _, _, ai = h.step(messages, browser_tools)
        assert _reused(ai) == _static_ends(h, messages, browser_tools)[1]
        assert _source(ai) == "ram"


# ── A reuse point where the generation prompt starts ──────────────────────────


@ALL_KINDS
@pytest.mark.parametrize("kv_bits", [None, 4], ids=["fp", "kv4"])
@pytest.mark.parametrize("next_turn", [
    HumanMessage("Compare it with Q3 last year."),
    HumanMessage("[page loaded] Example Domain — More information..."),
], ids=["follow_up", "observation_as_user_message"])
def test_turn_after_a_direct_answer_reuses_everything_before_the_scaffold(
    monkeypatch, kinds, kv_bits, next_turn,
):
    h = _Sessions(monkeypatch, kinds, kv_bits=kv_bits)
    tools = _tools()
    first = next(_run())
    h.step(first, tools)
    after = first + [AIMessage("Checked the report.</think>Revenue grew 12%."), next_turn]
    full, _, ai = h.step(after, tools)
    turn_start = _turn_start(h, first, tools)
    diverged = _diverged(h, first, after, tools)
    # Like Qwen3.5, the re-rendered assistant turn keeps its header but drops
    # the reasoning block the generation prompt opened.
    assert diverged == turn_start + len("<assistant>")
    if "linear" in kinds:
        assert _reused(ai) == turn_start
    else:
        assert _reused(ai) == diverged
    assert _source(ai) == "session"
    assert ai.response_metadata["tokens_prefilled"] == len(full) - _reused(ai)


@ALL_KINDS
def test_agent_steps_keep_reusing_the_whole_previous_prompt(monkeypatch, kinds):
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    previous = None
    for messages in _run(steps=4):
        full, _, ai = h.step(messages, tools)
        if previous is not None:
            assert _reused(ai) == len(previous) - 1
            assert _source(ai) == "session"
        previous = full


def test_hybrid_instance_keeps_at_most_two_recurrent_snapshots(monkeypatch):
    # The turn start of the latest question and the latest prompt boundary:
    # ~50 MB each for Qwen3.8-9B's 24 GatedDeltaNet layers.
    h = _Sessions(monkeypatch, HYBRID)
    tools = _tools()
    for messages in _run(steps=4):
        h.step(messages, tools)
        assert len(h.llm._reuse_points) <= 2


# ── Where the reuse came from ────────────────────────────────────────────────


def test_metadata_and_log_say_where_the_reuse_came_from(monkeypatch, caplog):
    h = _Sessions(monkeypatch, HYBRID)
    tools = _tools()
    run = _run(steps=2)
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, _, ai = h.step(next(run), tools)
    assert _source(ai) == "none"
    assert f"{len(full)} total tokens, 0 reused, {len(full)} new" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, _, ai = h.step(next(run), tools)
    assert _source(ai) == "session"
    assert f"{_reused(ai)} reused from this session's cache" in caplog.text

    h.new_session()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, _, ai = h.step(next(_run(question="Anything new?")), tools)
    assert _source(ai) == "ram"
    assert f"{_reused(ai)} reused (static prefix) from the RAM store" in caplog.text
    meta = ai.response_metadata
    assert meta["tokens_from_cache"] + meta["tokens_prefilled"] == len(full)


# ── The real Qwen3.5 template (tokenizer + chat_template only, no weights) ────

_REAL_SNAPSHOT = sorted(glob.glob(str(
    Path.home() / ".cache/huggingface/hub/models--Foresee--Qwen3.8-9B-heretic-uncensored-4bit-MTPLX"
    / "snapshots/*/tokenizer.json"
)))


@pytest.fixture
def real_llm(monkeypatch):
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(_REAL_SNAPSHOT[0]).parent)
    monkeypatch.setattr(chat_mlx_text, "_load_or_reuse", lambda *_: ((_Model(), tokenizer, None), False))
    monkeypatch.setattr(chat_mlx_text, "_WARMED_UP", set())
    monkeypatch.setattr(ChatMLXText, "_warmup", lambda self: None)
    return lambda **kw: ChatMLXText(model_path="real/qwen3.5-template", **kw)


@pytest.mark.skipif(not _REAL_SNAPSHOT, reason="Qwen3.8-9B tokenizer not in the local HF cache")
@pytest.mark.parametrize("thinking, scaffold", [
    (False, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    (True, "<|im_start|>assistant\n<think>\n"),
], ids=["thinking_off", "thinking_on"])
def test_real_qwen35_template_diverges_three_tokens_after_the_turn_start(real_llm, thinking, scaffold):
    llm = real_llm(thinking=thinking)
    tools = _tools()
    first = [SystemMessage("You are OTTO. Session files: /sessions/s1."),
             HumanMessage("Summarise the quarterly report.")]
    prev = llm._tokenizer.encode(llm._to_prompt(first, tools=tools))
    turn_start = llm._turn_start(prev)
    assert llm._tokenizer.decode(prev[turn_start:]) == scaffold

    def diverged(after):
        new = llm._tokenizer.encode(llm._to_prompt(after, tools=tools))
        return next((i for i, (a, b) in enumerate(zip(prev, new)) if a != b), len(prev))

    # Follow-up: "<|im_start|>", "assistant", "\n" survive; the re-rendered
    # turn has no "<think>" block — before the old len(prompt) - 1 boundary.
    follow_up = first + [AIMessage("Revenue grew 12%."), HumanMessage("Compare with Q3.")]
    assert diverged(follow_up) == turn_start + 3 < len(prev) - 1
    # An agent step (tool result) keeps the whole scaffold: the boundary covers it.
    call = {"name": "read_file", "args": {"path": "/r.txt"}, "id": "c1", "type": "tool_call"}
    step = first + [AIMessage("", tool_calls=[call]),
                    ToolMessage("Q3 revenue: 12M.", tool_call_id="c1", name="read_file")]
    assert diverged(step) >= len(prev) - 1
