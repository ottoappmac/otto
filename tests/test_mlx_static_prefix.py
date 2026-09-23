"""A static tools + system prefix is prefilled once per process, not once per session.

OTTO builds a new ``ChatMLXText`` for every session, so each session's first
call started from an empty cache; and a new user message re-renders the
history (Qwen3.5 templates drop the reasoning of earlier turns), so a hybrid
model couldn't roll its own cache back either.  Both paid a full prefill of
the ~35k-token tool block and system prompt although neither had changed.
``ChatMLXText`` now snapshots those static prefixes into a process-wide store
(:mod:`chat_models.mlx._prefix_store`) and restores them into any instance
running the same loaded model.

Built on the harness of ``test_mlx_prefix_cache``: the real ``mlx_lm``
generation loop driving a tiny deterministic model on real ``ArraysCache`` /
``KVCache`` layers.  Every step asserts the same invariant — the cache handed
to the model equals a fresh prefill of the tokens it claims to hold — so a
restored snapshot must be exact, private to its instance, and never taken
from another model, cache setting or token sequence.
"""

from __future__ import annotations

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

from chat_models.mlx import _prefix_store, _shared, chat_mlx_text  # noqa: E402
from chat_models.mlx.chat_mlx_text import ChatMLXText  # noqa: E402
from tests.test_mlx_prefix_cache import (  # noqa: E402
    ALL_KINDS,
    HYBRID,
    PURE_KV,
    _CharTokenizer,
    _Harness,
)


class _QwenLikeTokenizer(_CharTokenizer):
    """Char-level stand-in for the Qwen3.5 chat template.

    Like the real one it opens the system turn with the tool block, then the
    system prompt; drops the reasoning of assistant turns before the last
    user query; and rejects a prompt without a user query, so a system-only
    render can't be used to find the static prefix.
    """

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, **_,
    ):
        queries = [i for i, m in enumerate(messages) if m["role"] == "user"]
        if not queries:
            raise ValueError("No user query found in messages.")
        text = ""
        if tools or messages[0]["role"] == "system":
            text += "<system>" + "".join(
                f"[{t['function']['name']}: {t['function']['description']}]" for t in tools or ()
            )
            if messages[0]["role"] == "system":
                text += "\n" + messages[0]["content"]
            text += "</system>"
        for i, m in enumerate(messages):
            content = m["content"]
            if m["role"] == "system":
                continue
            if m["role"] == "assistant":
                reasoning, _, answer = content.rpartition("</think>")
                content = f"<think>{reasoning}</think>{answer}" if i > queries[-1] else answer
            text += f"<{m['role']}>{content}</{m['role']}>"
        return text + "<assistant><think>" if add_generation_prompt else text


class _MergingTokenizer(_QwenLikeTokenizer):
    """Encodes a newline or ``>`` followed by a capital letter as one token.

    Both static prefixes of the prompts below end right before such a letter
    (the system prompt and the question start with one), so on their own they
    tokenize differently than inside the prompt.
    """

    def encode(self, text, add_special_tokens=True):
        ids, i = [], 0
        while i < len(text):
            if text[i] in "\n>" and text[i + 1:i + 2].isupper():
                ids.append((128 if text[i] == ">" else 154) + ord(text[i + 1]) - ord("A"))
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


def _tools(*names, words=6):
    """OpenAI-format tool schemas; the default set renders to ~1.3k tokens."""
    names = names or ("read_file", "write_file", "search", "execute", "ask_user")
    return [
        {"type": "function", "function": {
            "name": name,
            "description": f"Use {name} on the session workspace. " * words,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        }}
        for name in names
    ]


def _system(session="s1"):
    # Like OTTO's: static rules, then the session's own files dir and time.
    return SystemMessage(
        "You are OTTO. Use the tools. " * 3 + f"Session files: /sessions/{session}. Now: 18:42."
    )


def _run(session="s1", question="Summarise the quarterly report.", steps=1):
    """Prompts of one agent run: every step adds an action and its result."""
    messages = [_system(session), HumanMessage(question)]
    yield list(messages)
    for i in range(steps - 1):
        messages += [
            AIMessage(f"I need part {i}.</think>Action: read_file(part={i})"),
            ToolMessage(f"part {i}: " + "revenue grew " * 4, tool_call_id=f"call_{i}"),
        ]
        yield list(messages)


def _static_ends(h, messages, tools):
    """Token lengths of the tool block and of the system turn + user opener."""
    text = h.llm._to_prompt(messages, tools=tools)
    return text.index("\n") + 1, text.index("<user>") + len("<user>")


class _Sessions(_Harness):
    """Sessions — separate ``ChatMLXText`` instances — on one loaded model."""

    def __init__(self, monkeypatch, kinds, tokenizer=None, **llm_kwargs):
        super().__init__(
            monkeypatch, kinds, tokenizer=tokenizer or _QwenLikeTokenizer(), **llm_kwargs,
        )
        self.llm_kwargs = {"max_tokens": 16, **llm_kwargs}

    def new_session(self, **overrides):
        self.llm = ChatMLXText(
            model_path="test/tiny-lm",
            enable_prompt_cache=True,
            enable_system_prompt_cache=True,
            **{**self.llm_kwargs, **overrides},
        )
        return self.llm


def _reused(ai):
    return ai.response_metadata["tokens_from_cache"]


def _on_fresh_thread(fn, *args):
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(fn, *args).result()


@pytest.fixture(autouse=True)
def store(monkeypatch):
    """A fresh process-wide store for every test."""
    fresh = _prefix_store.PrefixSnapshotStore()
    monkeypatch.setattr(_prefix_store, "STATIC_PREFIXES", fresh)
    return fresh


# ── The store ─────────────────────────────────────────────────────────────────


class _Model:
    """Any object that can be weakly referenced stands in for a loaded model."""


def test_store_keeps_the_most_recently_used_entries():
    store = _prefix_store.PrefixSnapshotStore(max_entries=2)
    model = _Model()
    store.put(model, "kv4", [1, 2], ["a"])
    store.put(model, "kv4", [1, 3], ["b"])
    assert store.get(model, "kv4", [1, 2]) == ["a"]  # now the most recently used
    store.put(model, "kv4", [1, 4], ["c"])
    assert len(store) == 2
    assert store.get(model, "kv4", [1, 3]) is None
    assert store.get(model, "kv4", [1, 2]) == ["a"]
    assert store.get(model, "kv4", [1, 4]) == ["c"]


def test_store_matches_only_the_same_model_settings_and_tokens():
    store = _prefix_store.PrefixSnapshotStore()
    model = _Model()
    store.put(model, "kv4", [1, 2, 3], ["x"])
    assert store.get(model, "kv4", [1, 2, 3]) == ["x"]
    assert store.get(_Model(), "kv4", [1, 2, 3]) is None
    assert store.get(model, "fp", [1, 2, 3]) is None
    assert store.get(model, "kv4", [1, 2]) is None
    assert store.get(model, "kv4", [1, 2, 3, 4]) is None


def test_store_never_hands_a_dead_models_snapshot_to_a_new_model(monkeypatch):
    # A new object can get a dead one's id(): pretend every object does.
    monkeypatch.setattr(_prefix_store, "id", lambda obj: 1, raising=False)
    store = _prefix_store.PrefixSnapshotStore()
    old = _Model()
    store.put(old, "kv4", [1, 2], ["x"])
    del old
    gc.collect()
    assert store.get(_Model(), "kv4", [1, 2]) is None


def test_evicting_mlx_models_drops_the_static_prefixes(monkeypatch, store):
    h = _Sessions(monkeypatch, HYBRID)
    h.step(next(_run()), _tools())
    assert len(store) == 2
    monkeypatch.setattr(_shared, "_LOADED_MODELS", {})
    monkeypatch.setattr(_shared, "_WARMED_UP", set())
    _shared.evict_all_mlx_models()
    assert len(store) == 0


# ── Restoring static prefixes ─────────────────────────────────────────────────


@ALL_KINDS
@pytest.mark.parametrize("kv_bits", [None, 4], ids=["fp", "kv4"])
def test_first_call_prefills_and_stores_the_static_prefixes(
    monkeypatch, caplog, store, kinds, kv_bits,
):
    h = _Sessions(monkeypatch, kinds, kv_bits=kv_bits)
    tools = _tools()
    messages = next(_run())
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, cached, ai = h.step(messages, tools)
    tool_end, system_end = _static_ends(h, messages, tools)
    # The model got a cache holding exactly the static prefix (checked by
    # the harness) and was fed the rest; nothing came from a cache.
    assert cached == system_end
    meta = ai.response_metadata
    assert meta["tokens_from_cache"] == 0
    assert meta["tokens_prefilled"] == len(full)
    assert meta["prompt_tps"] > 0
    assert len(store) == 2
    for n in (tool_end, system_end):
        assert f"stored the {n}-token static prefix" in caplog.text


@ALL_KINDS
@pytest.mark.parametrize("kv_bits", [None, 4], ids=["fp", "kv4"])
def test_new_session_restores_the_static_prefix(monkeypatch, caplog, kinds, kv_bits):
    h = _Sessions(monkeypatch, kinds, kv_bits=kv_bits)
    tools = _tools()
    h.step(next(_run()), tools)
    h.new_session()
    messages = next(_run(question="What changed since yesterday?"))
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, cached, ai = h.step(messages, tools)
    _, system_end = _static_ends(h, messages, tools)
    assert cached == _reused(ai) == system_end
    meta = ai.response_metadata
    assert meta["tokens_prefilled"] == len(full) - system_end
    assert meta["cache_hit_ratio"] == round(system_end / len(full), 3)
    assert f"{len(full)} total tokens, {system_end} reused (static prefix)" in caplog.text


@ALL_KINDS
def test_new_session_with_its_own_system_prompt_restores_the_tool_block(monkeypatch, kinds):
    # OTTO's system prompt names the session's files dir and the current
    # time, so only the tool block — rendered first by Qwen3.5 — is shared.
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    h.step(next(_run(session="s1")), tools)
    h.new_session()
    messages = next(_run(session="s2"))
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[0]


@ALL_KINDS
def test_new_user_message_restores_the_static_prefix(monkeypatch, kinds):
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    for messages in _run(steps=3):
        h.step(messages, tools)
    previous = h.tokenizer.encode(h.llm._to_prompt(messages, tools=tools))
    # The run ended with an answer.  The follow-up question re-renders that
    # turn without its reasoning, so the prompt diverges from the cached one
    # right after the first question.
    follow_up = messages + [
        AIMessage("All parts read.</think>Revenue grew 12%."),
        HumanMessage("Compare it with Q3 last year."),
    ]
    full, _, ai = h.step(follow_up, tools)
    _, system_end = _static_ends(h, follow_up, tools)
    diverged = next(i for i, (a, b) in enumerate(zip(previous, full)) if a != b)
    assert system_end < diverged
    if "linear" in kinds:
        # A recurrent state can only return to a snapshot.
        assert _reused(ai) == system_end
    else:
        # KV layers trim back to the divergence, past the static prefix.
        assert _reused(ai) == diverged


@ALL_KINDS
def test_agent_steps_after_a_restore_reuse_the_whole_previous_prompt(monkeypatch, caplog, kinds):
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    h.step(next(_run(session="s1")), tools)
    h.new_session()
    previous = None
    for messages in _run(session="s2", steps=3):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
            full, _, ai = h.step(messages, tools)
        if previous is not None:
            assert _reused(ai) >= len(previous) - 1
            assert "(static prefix)" not in caplog.text
        previous = full


def test_least_recently_used_static_prefix_is_evicted(monkeypatch, store):
    h = _Sessions(monkeypatch, HYBRID)
    tools = _tools()
    h.step(next(_run(session="s1")), tools)  # stores the tool block and s1's system turn
    h.new_session()
    h.step(next(_run(session="s2")), tools)  # restores the tool block, stores s2's system turn
    assert len(store) == 2
    # s1's system turn was evicted; the tool block, just used, wasn't.
    h.new_session()
    messages = next(_run(session="s1", question="Anything new?"))
    _, _, ai = h.step(messages, tools)
    tool_end, _ = _static_ends(h, messages, tools)
    assert _reused(ai) == tool_end
    # ...and bringing s1's back evicted s2's.
    h.new_session()
    _, _, ai = h.step(next(_run(session="s2", question="Anything new?")), tools)
    assert _reused(ai) == tool_end


@ALL_KINDS
def test_other_tools_never_reuse_a_static_prefix(monkeypatch, kinds):
    h = _Sessions(monkeypatch, kinds)
    h.step(next(_run()), _tools())
    h.new_session()
    # Same system prompt, only the last tool differs: the prompts share all
    # but the end of the tool block.
    other = _tools("read_file", "write_file", "search", "execute", "ask_human")
    _, _, ai = h.step(next(_run()), other)
    assert _reused(ai) == 0


def test_other_cache_settings_or_models_never_reuse_a_static_prefix(monkeypatch):
    tools = _tools()
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    h.step(next(_run()), tools)
    h.new_session(kv_bits=None)
    _, _, ai = h.step(next(_run()), tools)
    assert _reused(ai) == 0
    # Another loaded model with the same layout and prompt.
    other = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    _, _, ai = other.step(next(_run()), tools)
    assert _reused(ai) == 0


@pytest.mark.parametrize("kinds", [PURE_KV, HYBRID], ids=["pure_kv", "hybrid"])
def test_restored_snapshots_are_private_copies(monkeypatch, kinds):
    # Sessions write into their restored caches; the stored snapshot and the
    # other sessions' caches must not see it (the harness checks each step).
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    first_run = _run(session="s1", steps=2)
    h.step(next(first_run), tools)
    first = h.llm
    for session in ("s2", "s3"):
        h.new_session()
        for messages in _run(session=session, steps=2):
            h.step(messages, tools)
    h.llm = first
    previous = h.tokenizer.encode(first._to_prompt(next(_run(session="s1")), tools=tools))
    _, _, ai = h.step(next(first_run), tools)
    assert _reused(ai) >= len(previous) - 1


@ALL_KINDS
def test_sessions_on_different_threads(monkeypatch, kinds):
    # ``_agenerate`` runs every call via ``asyncio.to_thread``; MLX can't
    # evaluate a lazy array on another thread than the one that built it.
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    _on_fresh_thread(h.step, next(_run(session="s1")), tools)
    h.new_session()
    for messages in _run(session="s2", steps=2):
        _on_fresh_thread(h.step, messages, tools)
    h.new_session()
    messages = next(_run(session="s1", question="Anything new?"))
    _, _, ai = _on_fresh_thread(h.step, messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[0]


# ── What is not stored ────────────────────────────────────────────────────────


def test_prompts_without_tools_leave_the_store_alone(monkeypatch, store):
    # Title generation, memory ranking and extraction bind no tools; they
    # must not evict the agents' static prefixes.
    h = _Sessions(monkeypatch, HYBRID)
    tools = _tools()
    h.step(next(_run()), tools)
    utility = SystemMessage("Generate a concise title for the request. " * 40)
    h.new_session()
    h.step([utility, HumanMessage("Summarise the quarterly report.")])
    h.new_session()
    _, _, ai = h.step([utility, HumanMessage("Plan a trip.")])
    assert _reused(ai) == 0
    assert len(store) == 2
    h.new_session()
    messages = next(_run(question="Anything new?"))
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[1]


def test_short_static_prefixes_are_not_stored(monkeypatch, store):
    h = _Sessions(monkeypatch, HYBRID)
    h.step([SystemMessage("Be brief."), HumanMessage("Hi there.")], _tools(words=1))
    assert len(store) == 0


@pytest.mark.parametrize("kinds", [PURE_KV, HYBRID], ids=["pure_kv", "hybrid"])
def test_static_prefix_over_the_cache_budget_is_not_stored(monkeypatch, store, kinds):
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    messages = next(_run())
    h.llm.prompt_cache_max_tokens = _static_ends(h, messages, tools)[0] - 1
    h.step(messages, tools)
    assert len(store) == 0


@pytest.mark.parametrize("kinds", [PURE_KV, HYBRID], ids=["pure_kv", "hybrid"])
def test_cache_rebuilt_for_the_budget_restores_the_static_prefix(monkeypatch, caplog, kinds):
    # The prompt outgrows the budget, so the instance's cache is rebuilt
    # after the call; the next step still gets the static prefix back.
    h = _Sessions(monkeypatch, kinds)
    tools = _tools()
    run = _run(steps=2)
    messages = next(run)
    _, system_end = _static_ends(h, messages, tools)
    h.llm.prompt_cache_max_tokens = system_end + 5
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        h.step(messages, tools)
    assert "rebuilt" in caplog.text
    _, _, ai = h.step(next(run), tools)
    assert _reused(ai) == system_end


def test_static_prefix_that_tokenizes_differently_in_context_is_not_used(monkeypatch, store):
    h = _Sessions(monkeypatch, HYBRID, tokenizer=_MergingTokenizer())
    tools = _tools()
    h.step(next(_run()), tools)
    assert len(store) == 0
    h.new_session()
    _, _, ai = h.step(next(_run()), tools)
    assert _reused(ai) == 0


# ── Metadata ──────────────────────────────────────────────────────────────────


def test_prompt_throughput_counts_the_static_prefill(monkeypatch):
    # 300 static-prefix tokens took 3 s; stream_generate then prefilled 100
    # more at 50 tok/s (2 s): 400 tokens in 5 s.
    h = _Sessions(monkeypatch, HYBRID)
    last = SimpleNamespace(
        prompt_tokens=100, prompt_tps=50.0, generation_tokens=4,
        generation_tps=20.0, peak_memory=1.0,
    )
    meta = h.llm._build_response_metadata(last, 0, 300, 3.0)
    assert meta["tokens_prefilled"] == 400
    assert meta["prompt_tps"] == 80.0
