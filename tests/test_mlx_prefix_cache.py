"""Prefix-cache reuse in ``ChatMLXText`` must never hand the model a stale cache.

Hybrid models (Qwen3.5 / Qwen3-Next: GatedDeltaNet linear-attention layers
backed by a non-trimmable ``ArraysCache`` plus full-attention ``KVCache``
layers) used to get only their KV layers trimmed back to the common prefix,
while the recurrent state kept every token processed so far — prompt *and*
generated tokens.  Every agent step >= 2 then ran on a corrupted context.

These tests drive the real ``mlx_lm.stream_generate`` / ``generate_step``
(chunked prefill, progress callback, KV quantisation) with a tiny
deterministic stand-in model whose cache layers are real ``mlx_lm`` cache
objects recording which tokens they processed.  A spy captures the cache the
moment generation starts; the invariant is that it equals a fresh cache that
processed exactly the reused prefix of the new prompt.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache  # noqa: E402
from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402

from chat_models.mlx import chat_mlx_text  # noqa: E402
from chat_models.mlx.chat_mlx_text import ChatMLXText  # noqa: E402
from chat_models.mlx_turbo.chat import TurboMLXChat  # noqa: E402

_VOCAB = 256
_EOS = 0
_FIRST_GENERATED = 200  # ASCII prompts never encode to ids >= 200
_HEAD_DIM = 64  # kv_bits quantisation needs head_dim divisible by the group size

PURE_KV = ("attn", "attn", "attn")
HYBRID = ("linear", "linear", "attn", "linear", "attn")  # Qwen3.5-style interleave
RECURRENT = ("linear", "linear")  # Mamba-style: no KV layer, no offset at all


class _CharTokenizer:
    """One token per ASCII character; the chat template is append-only."""

    eos_token_id = _EOS
    bos_token = None
    chat_template = "fake template"
    clean_up_tokenization_spaces = False

    def get_vocab(self):
        return {}

    def encode(self, text, add_special_tokens=True):
        return [ord(ch) for ch in text]

    def decode(self, ids, **_):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        # The system prompt leads untagged, so prompts of agents with
        # different system prompts share no prefix at all.
        text = "".join(
            m["content"] if m["role"] == "system" else f"<{m['role']}>{m['content']}</{m['role']}>"
            for m in messages
        )
        return text + "<assistant>" if add_generation_prompt else text


class _TinyLM:
    """Deterministic stand-in for an ``mlx_lm`` model with real cache layers.

    ``linear`` layers keep an ``ArraysCache`` whose state is every token folded
    in so far — like a GatedDeltaNet recurrent state it can't be trimmed.
    ``attn`` layers write each token id into a real ``KVCache`` and, like
    attention, the logits depend on what they fetch back.  Each reply is one
    generated token (which depends on the recurrent state) followed by EOS.
    """

    def __init__(self, kinds):
        self.kinds = kinds
        self.layers = list(kinds)

    def make_cache(self):
        return [ArraysCache(size=1) if k == "linear" else KVCache() for k in self.kinds]

    def __call__(self, inputs, cache=None, input_embeddings=None):
        tokens = inputs.reshape(-1).tolist()
        kv = mx.array(tokens, dtype=mx.float32)[:, None] + mx.arange(_HEAD_DIM) / (2 * _HEAD_DIM)
        kv = kv.reshape(1, 1, len(tokens), _HEAD_DIM)
        context, attended = 0, mx.array(0.0)
        for kind, layer in zip(self.kinds, cache):
            if kind == "linear":
                seen = [] if layer[0] is None else layer[0].tolist()
                layer[0] = mx.array(seen + tokens, dtype=mx.int32)
                context = sum(seen + tokens)
            else:
                fetched = layer.update_and_fetch(kv, kv)
                attended = attended + sum(a.sum().astype(mx.float32) for _, a in tree_flatten(fetched))
        nxt = _EOS if tokens[-1] >= _FIRST_GENERATED else _FIRST_GENERATED + context % 8
        logits = 10.0 * (mx.arange(_VOCAB) == nxt) + 0.0 * attended
        return mx.broadcast_to(logits, (1, len(tokens), _VOCAB))


def _processed_tokens(cache):
    """Per layer, the token ids that layer has processed, in order."""
    out = []
    for layer in cache:
        if isinstance(layer, ArraysCache):
            out.append([] if layer[0] is None else layer[0].tolist())
        elif layer.keys is None:
            out.append([])
        else:
            keys = layer.state[0]
            if isinstance(layer, QuantizedKVCache):
                keys = mx.dequantize(*keys, group_size=layer.group_size, bits=layer.bits)
            out.append([round(v) for v in keys[0, 0, :, 0].tolist()])
    return out


def _fresh_prefill(model, tokens):
    cache = model.make_cache()
    if tokens:
        model(mx.array(tokens)[None], cache=cache)
    return _processed_tokens(cache)


class _Harness:
    """A ``ChatMLXText`` wired to ``_TinyLM`` with a spy on ``stream_generate``."""

    def __init__(self, monkeypatch, kinds, cls=ChatMLXText, tokenizer=None, **llm_kwargs):
        self.model = _TinyLM(kinds)
        self.tokenizer = TokenizerWrapper(tokenizer or _CharTokenizer())
        monkeypatch.setattr(
            chat_mlx_text, "_load_or_reuse",
            lambda *_: ((self.model, self.tokenizer, None), False),
        )
        monkeypatch.setattr(chat_mlx_text, "_WARMED_UP", set())
        monkeypatch.setattr(ChatMLXText, "_warmup", lambda self: None)

        self.starts = []
        real_stream_generate = mlx_lm.stream_generate

        def spy(model, tokenizer, prompt, **kwargs):
            fed = tokenizer.encode(prompt) if isinstance(prompt, str) else list(prompt)
            self.starts.append((fed, _processed_tokens(kwargs["prompt_cache"])))
            yield from real_stream_generate(model, tokenizer, prompt, **kwargs)

        monkeypatch.setattr(mlx_lm, "stream_generate", spy)
        llm_kwargs.setdefault("max_tokens", 16)
        self.llm = cls(
            model_path="test/tiny-lm",
            enable_prompt_cache=True,
            enable_system_prompt_cache=True,
            **llm_kwargs,
        )

    def step(self, messages, tools=None):
        """Generate once; assert the cache handed to the model was consistent.

        Returns ``(prompt_tokens, reused, ai_message)``.
        """
        ai = self.llm.invoke(messages, tools=tools)
        full = self.tokenizer.encode(self.llm._to_prompt(messages, tools=tools))
        fed, cache_at_start = self.starts[-1]
        reused = len(full) - len(fed)
        assert fed == full[reused:], "the model must be fed the suffix of the prompt"
        expected = _fresh_prefill(self.model, full[:reused])
        for i, (got, want) in enumerate(zip(cache_at_start, expected)):
            assert got == want, (
                f"call {len(self.starts)}, layer {i} ({self.model.kinds[i]}): the cache "
                f"handed to the model holds {len(got)} tokens ending {got[-3:]}, a fresh "
                f"prefill of the {reused}-token reused prefix holds {len(want)} ending {want[-3:]}"
            )
        return full, reused, ai


def _from_cache(ai):
    """Tokens the prompt cache supplied (the harness's ``reused`` also counts
    tokens prefilled before ``stream_generate``, e.g. up to a turn start)."""
    return ai.response_metadata["tokens_from_cache"]


def _agent_loop(steps, system_repeats=3):
    """Append-only history of one agent run: action, tool result, repeat."""
    messages = [
        SystemMessage("You are OTTO. Tools: read_file, search. " * system_repeats),
        HumanMessage("Summarise the quarterly report."),
    ]
    yield list(messages)
    for i in range(steps - 1):
        messages += [
            AIMessage(f"Action: read_file(part={i})"),
            ToolMessage(f"part {i}: " + "revenue grew " * 4, tool_call_id=f"call_{i}"),
        ]
        yield list(messages)


KINDS = pytest.mark.parametrize("kinds", [PURE_KV, HYBRID], ids=["pure_kv", "hybrid"])
ALL_KINDS = pytest.mark.parametrize(
    "kinds", [PURE_KV, HYBRID, RECURRENT], ids=["pure_kv", "hybrid", "recurrent"],
)


@ALL_KINDS
@pytest.mark.parametrize("kv_bits", [None, 4], ids=["fp", "kv4"])
def test_agent_loop_reuses_a_consistent_prefix(monkeypatch, kinds, kv_bits):
    h = _Harness(monkeypatch, kinds, kv_bits=kv_bits)
    previous = None
    for messages in _agent_loop(4):
        full, reused, _ = h.step(messages)
        if previous is not None:
            # Append-only history: only the previous prompt's last token (the
            # prompt boundary) and the new turn need prefilling.
            assert reused >= len(previous) - 1
        previous = full


@ALL_KINDS
def test_long_prompt_prefilled_in_chunks(monkeypatch, kinds):
    # mlx_lm prefills 2048 tokens at a time and reports progress after each
    # chunk; only the report at the prompt boundary may be recorded.
    h = _Harness(monkeypatch, kinds)
    previous = None
    for messages in _agent_loop(3, system_repeats=120):  # ~4.9k tokens, 3 chunks
        full, reused, _ = h.step(messages)
        if previous is not None:
            assert reused >= len(previous) - 1
        previous = full


@ALL_KINDS
def test_consecutive_calls_on_different_threads(monkeypatch, kinds):
    # ``_agenerate`` runs every call via ``asyncio.to_thread``, and MLX can't
    # evaluate a lazy array on a thread other than the one that built it —
    # so nothing unevaluated may be left in the cache between calls.
    h = _Harness(monkeypatch, kinds)
    for messages in _agent_loop(3):
        with ThreadPoolExecutor(max_workers=1) as fresh_thread:
            fresh_thread.submit(h.step, messages).result()


@ALL_KINDS
def test_diverging_history_never_reuses_stale_state(monkeypatch, kinds):
    h = _Harness(monkeypatch, kinds)
    loop = _agent_loop(2)
    first, _, _ = h.step(next(loop))
    h.step(next(loop))
    # New user turn that rewrites history (e.g. summarisation): the prompts
    # now diverge inside the previous prompt.
    rewritten = [SystemMessage("You are OTTO. Tools: read_file, search. " * 3),
                 HumanMessage("Summarise the annual report instead.")]
    full, _, ai = h.step(rewritten)
    common = next(i for i, (a, b) in enumerate(zip(first, full)) if a != b)
    if "linear" in kinds:
        # A recurrent state can't be rolled back to an arbitrary point.
        assert _from_cache(ai) == 0
    else:
        assert _from_cache(ai) == common
    # ...and reuse resumes on the next append-only step.
    _, reused, _ = h.step(rewritten + [AIMessage("Action: search()"),
                                       ToolMessage("no hits", tool_call_id="c")])
    assert reused >= len(full) - 1


@ALL_KINDS
def test_resending_the_same_prompt_feeds_its_last_token(monkeypatch, kinds):
    h = _Harness(monkeypatch, kinds)
    messages = next(_agent_loop(1))
    full, _, _ = h.step(messages)
    _, reused, _ = h.step(messages)
    assert reused == len(full) - 1


@ALL_KINDS
def test_unrelated_prompt_starts_from_an_empty_cache(monkeypatch, kinds):
    # Another agent (different system prompt) on the same instance: nothing
    # is reusable, and the previous context must not leak into its prompt.
    h = _Harness(monkeypatch, kinds)
    h.step(next(_agent_loop(1)))
    _, _, ai = h.step([SystemMessage("Memory ranker: score each note."),
                       HumanMessage("Which notes matter?")])
    assert _from_cache(ai) == 0


@KINDS
def test_budget_only_drops_the_generated_tail(monkeypatch, kinds, caplog):
    h = _Harness(monkeypatch, kinds)
    loop = _agent_loop(2)
    messages = next(loop)
    prompt_len = len(h.tokenizer.encode(h.llm._to_prompt(messages)))
    # Room for the prompt, but not for prompt + generated tokens.
    h.llm.prompt_cache_max_tokens = prompt_len
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, _, _ = h.step(messages)
    assert "exceeded budget" in caplog.text
    _, reused, _ = h.step(next(loop))
    assert reused >= len(full) - 1


@KINDS
def test_budget_smaller_than_the_prompt_rebuilds_the_cache(monkeypatch, kinds, caplog):
    h = _Harness(monkeypatch, kinds)
    loop = _agent_loop(2)
    messages = next(loop)
    prompt_len = len(h.tokenizer.encode(h.llm._to_prompt(messages)))
    h.llm.prompt_cache_max_tokens = prompt_len // 2
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        h.step(messages)
    assert "rebuilt" in caplog.text
    _, reused, _ = h.step(next(loop))
    assert reused == 0


@ALL_KINDS
def test_metadata_reports_true_reused_and_prefilled_counts(monkeypatch, kinds, caplog):
    h = _Harness(monkeypatch, kinds)
    loop = _agent_loop(2)
    h.step(next(loop))
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, reused, ai = h.step(next(loop))
    meta = ai.response_metadata
    assert meta["tokens_from_cache"] == reused
    assert meta["tokens_prefilled"] == len(full) - reused
    assert meta["cache_hit_ratio"] == round(reused / len(full), 3)
    assert f"{len(full)} total tokens, {reused} reused" in caplog.text


class _FakeSSDStore:
    """Stands in for ``SSDPrefixStore``: always 'finds' one saved cache."""

    def __init__(self, cache, prefix_len):
        self.cache, self.prefix_len = cache, prefix_len

    def find_longest_match(self, tokens):
        return self.prefix_len, "/tmp/fake-entry.safetensors"

    def load(self, path):
        return self.cache, {}

    def find_best_common_prefix(self, tokens):
        return 0

    def save(self, tokens, cache, extra_meta=None):
        pass


@pytest.mark.parametrize(
    "kinds, reusable", [(PURE_KV, True), (HYBRID, False)], ids=["pure_kv", "hybrid"],
)
def test_ssd_prime_only_reuses_caches_it_can_roll_back_exactly(monkeypatch, kinds, reusable):
    h = _Harness(monkeypatch, kinds, cls=TurboMLXChat, turbo_level="cache")
    messages = next(_agent_loop(1))
    full = h.tokenizer.encode(h.llm._to_prompt(messages))
    prefix_len = len(full) - 5
    # Saved after a turn: the snapshot holds the prefix plus generated tokens.
    saved = h.model.make_cache()
    h.model(mx.array(full[:prefix_len] + [_FIRST_GENERATED + 1, _EOS])[None], cache=saved)
    mx.eval([c.state for c in saved])  # built here, used on the turbo MLX thread
    h.llm.turbo_level = "ssd"
    h.llm._ssd_store = _FakeSSDStore(saved, prefix_len)

    _, _, ai = h.step(messages)

    assert _from_cache(ai) == (prefix_len if reusable else 0)
