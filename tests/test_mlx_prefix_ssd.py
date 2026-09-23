"""Static-prefix snapshots survive a backend restart on SSD.

After a restart every agent family used to pay a full cold prefill of its
~35k-token tool block and system turn again (~130 s each).  Static snapshots
that can be hit again — an agent's tool block, and any other from its second
use on — are now also written to ``~/Library/Caches/otto/prefix``
(``_prefix_disk``) with ``mlx_lm``'s ``save_prompt_cache``, keyed by the
loaded model weights, the KV settings and the exact token ids; a RAM-store
miss loads the exact-length snapshot from there.  Every use touches a file,
so the directory's LRU cap evicts what nobody uses.

The unit tests check the file format itself: a hybrid snapshot (4-bit
``QuantizedKVCache`` + ``ArraysCache`` with empty slots) must come back
bit-exact and give the same next-step logits; corrupt or partial files are
deleted and the prompt is prefilled instead.  The integration tests reuse the
``test_mlx_prefix_cache`` harness, whose invariant — the cache handed to the
model equals a fresh prefill of the tokens it claims to hold — also holds for
a snapshot loaded from disk.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache  # noqa: E402
from mlx_lm.tokenizer_utils import TokenizerWrapper  # noqa: E402

from chat_models.mlx import _prefix_disk, _prefix_store, _shared, chat_mlx_text  # noqa: E402
from chat_models.mlx.chat_mlx_text import ChatMLXText, _clone_cache_layer  # noqa: E402
from tests.test_mlx_prefix_cache import ALL_KINDS, HYBRID, _TinyLM  # noqa: E402
from tests.test_mlx_static_prefix import (  # noqa: E402
    _QwenLikeTokenizer,
    _reused,
    _run,
    _Sessions,
    _static_ends,
    _tools,
)

_WEIGHTS = "tiny-lm-weights-v1"


@pytest.fixture(autouse=True)
def disk(monkeypatch, tmp_path):
    """A fresh RAM store and an SSD tier in a temporary directory."""
    monkeypatch.setattr(_prefix_store, "STATIC_PREFIXES", _prefix_store.PrefixSnapshotStore())
    store = _prefix_disk.DiskPrefixStore(tmp_path / "prefix")
    monkeypatch.setattr(_prefix_disk, "SSD_PREFIXES", store)
    _loaded_from(monkeypatch, _WEIGHTS)
    return store


def _loaded_from(monkeypatch, weights):
    """Every loaded model now counts as loaded from the files fingerprinted *weights*."""
    monkeypatch.setattr(chat_mlx_text, "weights_fingerprint", lambda model: weights)


def _files(disk):
    return sorted(disk.directory.glob("*.safetensors")) if disk.directory.exists() else []


def _restart(monkeypatch, kinds, weights=_WEIGHTS, **llm_kwargs):
    """A new backend process: a new model object and empty stores; the SSD directory stays."""
    monkeypatch.setattr(_prefix_store, "STATIC_PREFIXES", _prefix_store.PrefixSnapshotStore())
    disk = _prefix_disk.SSD_PREFIXES
    monkeypatch.setattr(_prefix_disk, "SSD_PREFIXES", _prefix_disk.DiskPrefixStore(disk.directory, disk.max_bytes))
    _loaded_from(monkeypatch, weights)
    return _Sessions(monkeypatch, kinds, **llm_kwargs)


# ── The file format ───────────────────────────────────────────────────────────

_TOKENS = 70


def _hybrid_snapshot():
    """A snapshot as the RAM store holds it: evaluated clones of every layer."""
    keys = mx.random.normal((1, 2, _TOKENS, 64), key=mx.random.key(1)).astype(mx.bfloat16)
    values = mx.random.normal((1, 2, _TOKENS, 64), key=mx.random.key(2)).astype(mx.bfloat16)
    quantized = KVCache()
    quantized.update_and_fetch(keys, values)
    quantized = quantized.to_quantized(group_size=64, bits=4)
    plain = KVCache()
    plain.update_and_fetch(values, keys)
    leading_hole = ArraysCache(size=2)  # [None, state]
    leading_hole[1] = mx.random.normal((1, 3, 32), key=mx.random.key(3)).astype(mx.bfloat16)
    middle_hole = ArraysCache(size=3)  # [conv, None, ssm]
    middle_hole[0] = mx.random.normal((1, 3, 32), key=mx.random.key(4)).astype(mx.bfloat16)
    middle_hole[2] = mx.random.normal((1, 4, 16, 16), key=mx.random.key(5))
    return [_clone_cache_layer(c) for c in (leading_hole, quantized, middle_hole, plain)]


def _next_logits(cache):
    """One decode step reading every layer: attention over the (dequantized) KV, recurrent sums."""
    x = mx.random.normal((1, 2, 1, 64), key=mx.random.key(6)).astype(mx.bfloat16)
    features = []
    for layer in cache:
        if isinstance(layer, ArraysCache):
            features += [mx.sum(a.astype(mx.float32)) for a in layer.state if a is not None]
            continue
        keys, values = layer.update_and_fetch(x, x)
        if isinstance(layer, QuantizedKVCache):
            keys = mx.dequantize(*keys, group_size=layer.group_size, bits=layer.bits)
            values = mx.dequantize(*values, group_size=layer.group_size, bits=layer.bits)
        scores = x.astype(mx.float32) @ keys.astype(mx.float32).swapaxes(-1, -2) / 8.0
        features.append(mx.sum(mx.softmax(scores, axis=-1) @ values.astype(mx.float32)))
    weights = mx.random.normal((256, len(features)), key=mx.random.key(7))
    return weights @ mx.stack(features)


def _assert_bit_exact(a_layers, b_layers):
    assert len(a_layers) == len(b_layers)
    for a, b in zip(a_layers, b_layers):
        assert type(a) is type(b)
        assert tuple(a.meta_state) == tuple(b.meta_state)
        a_flat = tree_flatten(a.state, is_leaf=lambda x: x is None)
        b_flat = tree_flatten(b.state, is_leaf=lambda x: x is None)
        assert [k for k, _ in a_flat] == [k for k, _ in b_flat]
        for (_, x), (_, y) in zip(a_flat, b_flat):
            if x is None or y is None:
                assert x is None and y is None
                continue
            assert x.dtype == y.dtype and x.shape == y.shape
            assert mx.array_equal(x, y).item()


def test_saved_hybrid_snapshot_loads_bit_exact_with_the_same_next_logits(disk):
    snapshot = _hybrid_snapshot()
    assert disk.save("a" * 64, snapshot, _TOKENS)
    loaded = disk.load("a" * 64, _TOKENS, len(snapshot))
    assert loaded is not None
    _assert_bit_exact(snapshot, loaded)
    assert loaded[0][0] is None and loaded[2][1] is None
    in_ram = _next_logits([_clone_cache_layer(c) for c in snapshot])
    from_disk = _next_logits([_clone_cache_layer(c) for c in loaded])
    assert mx.array_equal(in_ram, from_disk).item()


@pytest.mark.parametrize("damage", ["truncated", "garbage", "empty"])
def test_corrupt_or_partial_file_is_deleted(disk, damage):
    snapshot = _hybrid_snapshot()
    disk.save("b" * 64, snapshot, _TOKENS)
    (path,) = _files(disk)
    data = path.read_bytes()
    path.write_bytes({"truncated": data[: len(data) // 2], "garbage": b"\x00junk" * 64, "empty": b""}[damage])
    assert disk.load("b" * 64, _TOKENS, len(snapshot)) is None
    assert not path.exists()


def test_snapshot_of_other_tokens_or_layers_is_rejected(disk):
    snapshot = _hybrid_snapshot()
    disk.save("c" * 64, snapshot, _TOKENS)
    assert disk.load("c" * 64, _TOKENS + 1, len(snapshot)) is None
    disk.save("c" * 64, snapshot, _TOKENS)
    assert disk.load("c" * 64, _TOKENS, len(snapshot) + 1) is None


def test_missing_file_is_a_plain_miss(disk):
    assert disk.load("d" * 64, _TOKENS, 4) is None


def test_directory_is_capped_least_recently_used_first(tmp_path):
    snapshot = _hybrid_snapshot()
    probe = _prefix_disk.DiskPrefixStore(tmp_path / "probe")
    probe.save("0" * 64, snapshot, _TOKENS)
    size = _files(probe)[0].stat().st_size
    disk = _prefix_disk.DiskPrefixStore(tmp_path / "capped", max_bytes=int(size * 2.5))
    now = time.time()
    for i, key in enumerate(("1" * 64, "2" * 64)):
        disk.save(key, snapshot, _TOKENS)
        os.utime(disk.directory / f"{key}.safetensors", (now - 100 + i, now - 100 + i))
    assert disk.load("1" * 64, _TOKENS, len(snapshot)) is not None  # now the most recently used
    disk.save("3" * 64, snapshot, _TOKENS)
    assert sorted(p.stem for p in _files(disk)) == ["1" * 64, "3" * 64]


def test_disabled_tier_stores_nothing(tmp_path):
    disk = _prefix_disk.DiskPrefixStore(None)
    assert disk.save("e" * 64, _hybrid_snapshot(), _TOKENS) is False
    assert disk.load("e" * 64, _TOKENS, 4) is None


def test_directory_comes_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(_prefix_disk.CACHE_DIR_ENV, raising=False)
    assert _prefix_disk.cache_dir_from_env() == Path.home() / "Library" / "Caches" / "otto" / "prefix"
    monkeypatch.setenv(_prefix_disk.CACHE_DIR_ENV, str(tmp_path / "elsewhere"))
    assert _prefix_disk.cache_dir_from_env() == tmp_path / "elsewhere"
    monkeypatch.setenv(_prefix_disk.CACHE_DIR_ENV, "")
    assert _prefix_disk.cache_dir_from_env() is None


def test_key_covers_weights_kv_settings_and_exact_tokens():
    key = _prefix_disk.snapshot_key
    base = key("weights", (4, 64), [1, 2, 3])
    assert base == key("weights", (4, 64), [1, 2, 3])
    assert len({
        base,
        key("other weights", (4, 64), [1, 2, 3]),
        key("weights", (8, 64), [1, 2, 3]),
        key("weights", (4, 32), [1, 2, 3]),
        key("weights", (None, 64), [1, 2, 3]),
        key("weights", (4, 64), [1, 2]),
        key("weights", (4, 64), [1, 2, 4]),
    }) == 7


def test_model_fingerprint_follows_the_weight_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    weights = model / "model.safetensors"
    weights.write_bytes(b"w" * 10)
    first = _prefix_disk.model_fingerprint(str(model))
    assert first is not None and first == _prefix_disk.model_fingerprint(str(model))
    weights.write_bytes(b"w" * 11)
    resized = _prefix_disk.model_fingerprint(str(model))
    assert resized != first
    os.utime(weights, (1_000_000, 1_000_000))
    touched = _prefix_disk.model_fingerprint(str(model))
    assert touched not in (first, resized)
    (model / "config.json").write_text('{"num_hidden_layers": 2}')
    assert _prefix_disk.model_fingerprint(str(model)) not in (first, resized, touched)
    # No local weights, no fingerprint: nothing may be shared across restarts.
    assert _prefix_disk.model_fingerprint(str(tmp_path / "missing")) is None
    assert _prefix_disk.model_fingerprint("test/not-a-cached-repo") is None


def test_snapshots_are_keyed_by_the_weights_that_were_loaded(monkeypatch, tmp_path):
    # A catalog re-download replaces the files while the old weights stay
    # loaded: what new sessions prefill with them isn't the new weights' state.
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"v1" * 10)
    loaded = _prefix_disk.model_fingerprint(str(model_dir))
    model = _TinyLM(HYBRID)
    monkeypatch.setattr(mlx_lm, "load", lambda path: (model, TokenizerWrapper(_QwenLikeTokenizer())))
    monkeypatch.setattr(_shared, "_LOADED_MODELS", {})
    monkeypatch.setattr(_shared, "_WEIGHTS_FINGERPRINTS", {})
    monkeypatch.setattr(chat_mlx_text, "_WARMED_UP", set())
    monkeypatch.setattr(ChatMLXText, "_warmup", lambda self: None)
    monkeypatch.setattr(chat_mlx_text, "weights_fingerprint", _shared.weights_fingerprint)

    def session():
        return ChatMLXText(
            model_path=str(model_dir), enable_prompt_cache=True,
            enable_system_prompt_cache=True, kv_bits=4,
        )

    first = session()
    weights.write_bytes(b"v2" * 12)
    assert _prefix_disk.model_fingerprint(str(model_dir)) != loaded
    second = session()
    assert second._model is model  # reused, not loaded again
    tokens = list(range(1, 50))
    expected = _prefix_disk.snapshot_key(loaded, (4, 64), tokens)
    assert first._ssd_key(tokens) == second._ssd_key(tokens) == expected


# ── Across a restart ──────────────────────────────────────────────────────────


def _use_the_system_turn_again(h, tools):
    """A second session with the same system prompt: its system turn is written to SSD too."""
    h.new_session()
    h.step(next(_run(question="Anything new?")), tools)


@ALL_KINDS
@pytest.mark.parametrize("kv_bits", [None, 4], ids=["fp", "kv4"])
def test_restart_restores_the_static_prefix_from_ssd(monkeypatch, caplog, disk, kinds, kv_bits):
    h = _Sessions(monkeypatch, kinds, kv_bits=kv_bits)
    tools = _tools()
    h.step(next(_run()), tools)
    _use_the_system_turn_again(h, tools)
    assert len(_files(disk)) == 2  # the tool block and the system turn

    h = _restart(monkeypatch, kinds, kv_bits=kv_bits)
    messages = next(_run(question="What changed since yesterday?"))
    with caplog.at_level(logging.INFO, logger=chat_mlx_text.__name__):
        full, _, ai = h.step(messages, tools)
    _, system_end = _static_ends(h, messages, tools)
    assert _reused(ai) == system_end
    assert ai.response_metadata["prefix_cache_source"] == "ssd"
    assert ai.response_metadata["tokens_prefilled"] == len(full) - system_end
    assert f"{len(full)} total tokens, {system_end} reused (static prefix) from SSD" in caplog.text
    assert len(_files(disk)) == 2  # a hit writes nothing new

    # The next session loads it again: a snapshot loaded from SSD goes to its
    # session, not to the RAM store.
    h.new_session()
    _, _, ai = h.step(next(_run(question="Anything new?")), tools)
    assert _reused(ai) == system_end
    assert ai.response_metadata["prefix_cache_source"] == "ssd"


@pytest.mark.parametrize("change", [{"weights": "retrained"}, {"kv_bits": None}], ids=["weights", "kv_bits"])
def test_other_weights_or_kv_settings_never_load_a_snapshot(monkeypatch, disk, change):
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run()), tools)
    settings = {"weights": _WEIGHTS, "kv_bits": 4, **change}
    h = _restart(monkeypatch, HYBRID, weights=settings.pop("weights"), **settings)
    _, _, ai = h.step(next(_run()), tools)
    assert _reused(ai) == 0
    assert len(_files(disk)) == 2  # each wrote its own tool block


def test_corrupt_snapshots_are_deleted_and_prefilled_again(monkeypatch, caplog, disk):
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run()), tools)
    _use_the_system_turn_again(h, tools)
    tool_block, system_turn = sorted(_files(disk), key=lambda p: p.stat().st_size)
    tool_block.write_bytes(tool_block.read_bytes()[:-100])  # partial write
    system_turn.write_bytes(b"not a safetensors file")

    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    messages = next(_run(question="What changed since yesterday?"))
    with caplog.at_level(logging.WARNING, logger=_prefix_disk.__name__):
        full, _, ai = h.step(messages, tools)
    assert _reused(ai) == 0
    assert ai.response_metadata["tokens_prefilled"] == len(full)
    assert caplog.text.count("deleted") == 2
    # ...and they were written again — the tool block by the fresh prefill,
    # the system turn once used again — readable this time.
    _use_the_system_turn_again(h, tools)
    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[1]
    assert ai.response_metadata["prefix_cache_source"] == "ssd"


def test_unknown_weights_keep_snapshots_off_the_ssd(monkeypatch, disk):
    _loaded_from(monkeypatch, None)
    h = _Sessions(monkeypatch, HYBRID)
    h.step(next(_run()), _tools())
    assert _files(disk) == []


# ── What goes to SSD, and what stays there ────────────────────────────────────


def _tool_block_file(disk):
    return min(_files(disk), key=lambda p: p.stat().st_size)


def test_new_sessions_write_only_their_shared_tool_block(monkeypatch, disk):
    # OTTO's system turn names the session's files dir and the current time:
    # no other session, and no session after a restart, hits it again.
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    for session in ("s1", "s2", "s3"):
        h.new_session()
        h.step(next(_run(session=session)), tools)
    assert len(_files(disk)) == 1

    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    messages = next(_run(session="s4"))
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[0]
    assert ai.response_metadata["prefix_cache_source"] == "ssd"


@pytest.mark.parametrize("ram_evicted", [False, True], ids=["ram_hit", "prefilled_again"])
def test_a_system_turn_used_again_is_written(monkeypatch, disk, ram_evicted):
    # A subagent's system prompt is the same in every session: from its second
    # use on — found in RAM, or prefilled again once other agents pushed its
    # family out of RAM — it is worth a file.
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run()), tools)
    assert len(_files(disk)) == 1
    if ram_evicted:
        _prefix_store.STATIC_PREFIXES.clear()
    h.new_session()
    h.step(next(_run(question="Anything new?")), tools)
    assert len(_files(disk)) == 2

    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    messages = next(_run(question="What changed since yesterday?"))
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[1]
    assert ai.response_metadata["prefix_cache_source"] == "ssd"


def test_a_ram_hit_marks_its_ssd_file_most_recently_used(monkeypatch, disk):
    # The SSD tier evicts by mtime; a snapshot served from RAM is in use too.
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run(session="s1")), tools)
    tool_block = _tool_block_file(disk)
    os.utime(tool_block, (1_000_000, 1_000_000))
    inode = tool_block.stat().st_ino
    h.new_session()
    _, _, ai = h.step(next(_run(session="s2")), tools)
    assert ai.response_metadata["prefix_cache_source"] == "ram"
    assert tool_block.stat().st_mtime > time.time() - 60
    assert tool_block.stat().st_ino == inode  # touched, not written again


def test_the_tool_block_outlives_many_sessions_and_a_restart(monkeypatch, disk):
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run(session="s1")), tools)
    # Room for the tool block and about two system turns.
    disk.max_bytes = 4 * _tool_block_file(disk).stat().st_size
    for i in range(2, 8):
        h.new_session()
        h.step(next(_run(session=f"s{i}")), tools)

    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    messages = next(_run(session="s99"))
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[0]
    assert ai.response_metadata["prefix_cache_source"] == "ssd"


def test_snapshots_are_written_after_generation_outside_the_lock(monkeypatch, disk):
    # A write (~0.37 GB in the app) must not hold up this call's first token,
    # nor another agent waiting for MLX_GEN_LOCK.
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    real_save, calls = disk.save, []

    def save(key, layers, n_tokens):
        calls.append((chat_mlx_text.MLX_GEN_LOCK.locked(), len(h.starts)))
        return real_save(key, layers, n_tokens)

    monkeypatch.setattr(disk, "save", save)
    h.step(next(_run()), _tools())
    assert calls == [(False, 1)]  # the tool block, once the reply was generated


def test_a_file_evicted_by_another_process_while_loading_is_still_a_hit(monkeypatch, disk):
    # Two backends can share the directory; touching a file just evicted fails.
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run()), tools)
    real_read = _prefix_disk.DiskPrefixStore._read

    def read_then_evicted(path, *args):
        layers = real_read(path, *args)
        path.unlink()
        return layers

    monkeypatch.setattr(_prefix_disk.DiskPrefixStore, "_read", staticmethod(read_then_evicted))
    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    messages = next(_run(session="s2"))
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[0]


def test_a_snapshot_loaded_from_ssd_goes_to_its_session_not_to_the_ram_store(monkeypatch, disk):
    # It's the session's own copy; another one would sit in RAM although the
    # next session can load it again in ~0.1-0.3 s.
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run(session="s1")), tools)

    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    for session in ("s2", "s3"):
        h.new_session()
        messages = next(_run(session=session))
        full, _, ai = h.step(messages, tools)
        tool_end, system_end = _static_ends(h, messages, tools)
        assert _reused(ai) == tool_end
        assert ai.response_metadata["prefix_cache_source"] == "ssd"
        ram = _prefix_store.STATIC_PREFIXES
        assert ram.get(h.model, (4, 64), full[:tool_end]) is None
        assert ram.get(h.model, (4, 64), full[:system_end]) is not None  # prefilled here
