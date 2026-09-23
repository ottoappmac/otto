"""Static-prefix snapshots survive a backend restart on SSD.

After a restart every agent family used to pay a full cold prefill of its
~35k-token tool block and system turn again (~130 s each).  Newly prefilled
static snapshots are now also written to ``~/Library/Caches/otto/prefix``
(``_prefix_disk``) with ``mlx_lm``'s ``save_prompt_cache``, keyed by the
model weights, the KV settings and the exact token ids; a RAM-store miss
loads the exact-length snapshot from there.

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
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache  # noqa: E402

from chat_models.mlx import _prefix_disk, _prefix_store, chat_mlx_text  # noqa: E402
from chat_models.mlx.chat_mlx_text import _clone_cache_layer  # noqa: E402
from tests.test_mlx_prefix_cache import ALL_KINDS, HYBRID  # noqa: E402
from tests.test_mlx_static_prefix import _reused, _run, _Sessions, _static_ends, _tools  # noqa: E402

_REAL_FINGERPRINT = _prefix_disk.model_fingerprint
_WEIGHTS = "tiny-lm-weights-v1"


@pytest.fixture(autouse=True)
def disk(monkeypatch, tmp_path):
    """A fresh RAM store and an SSD tier in a temporary directory."""
    monkeypatch.setattr(_prefix_store, "STATIC_PREFIXES", _prefix_store.PrefixSnapshotStore())
    store = _prefix_disk.DiskPrefixStore(tmp_path / "prefix")
    monkeypatch.setattr(_prefix_disk, "SSD_PREFIXES", store)
    monkeypatch.setattr(_prefix_disk, "model_fingerprint", lambda path: _WEIGHTS)
    return store


def _files(disk):
    return sorted(disk.directory.glob("*.safetensors")) if disk.directory.exists() else []


def _restart(monkeypatch, kinds, weights=_WEIGHTS, **llm_kwargs):
    """A new backend process: a new model object and an empty RAM store; the SSD stays."""
    monkeypatch.setattr(_prefix_store, "STATIC_PREFIXES", _prefix_store.PrefixSnapshotStore())
    monkeypatch.setattr(_prefix_disk, "model_fingerprint", lambda path: weights)
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
    first = _REAL_FINGERPRINT(str(model))
    assert first is not None and first == _REAL_FINGERPRINT(str(model))
    weights.write_bytes(b"w" * 11)
    resized = _REAL_FINGERPRINT(str(model))
    assert resized != first
    os.utime(weights, (1_000_000, 1_000_000))
    assert _REAL_FINGERPRINT(str(model)) not in (first, resized)
    # No local weights, no fingerprint: nothing may be shared across restarts.
    assert _REAL_FINGERPRINT(str(tmp_path / "missing")) is None
    assert _REAL_FINGERPRINT("test/not-a-cached-repo") is None


# ── Across a restart ──────────────────────────────────────────────────────────


@ALL_KINDS
@pytest.mark.parametrize("kv_bits", [None, 4], ids=["fp", "kv4"])
def test_restart_restores_the_static_prefix_from_ssd(monkeypatch, caplog, disk, kinds, kv_bits):
    h = _Sessions(monkeypatch, kinds, kv_bits=kv_bits)
    tools = _tools()
    h.step(next(_run()), tools)
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

    # Loaded once, then served from RAM.
    h.new_session()
    _, _, ai = h.step(next(_run(question="Anything new?")), tools)
    assert _reused(ai) == system_end
    assert ai.response_metadata["prefix_cache_source"] == "ram"


@pytest.mark.parametrize("change", [{"weights": "retrained"}, {"kv_bits": None}], ids=["weights", "kv_bits"])
def test_other_weights_or_kv_settings_never_load_a_snapshot(monkeypatch, disk, change):
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run()), tools)
    settings = {"weights": _WEIGHTS, "kv_bits": 4, **change}
    h = _restart(monkeypatch, HYBRID, weights=settings.pop("weights"), **settings)
    _, _, ai = h.step(next(_run()), tools)
    assert _reused(ai) == 0
    assert len(_files(disk)) == 4


def test_corrupt_snapshots_are_deleted_and_prefilled_again(monkeypatch, caplog, disk):
    h = _Sessions(monkeypatch, HYBRID, kv_bits=4)
    tools = _tools()
    h.step(next(_run()), tools)
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
    # ...and the fresh prefill wrote them again, readable this time.
    h = _restart(monkeypatch, HYBRID, kv_bits=4)
    _, _, ai = h.step(messages, tools)
    assert _reused(ai) == _static_ends(h, messages, tools)[1]
    assert ai.response_metadata["prefix_cache_source"] == "ssd"


def test_unknown_weights_keep_snapshots_off_the_ssd(monkeypatch, disk):
    monkeypatch.setattr(_prefix_disk, "model_fingerprint", lambda path: None)
    h = _Sessions(monkeypatch, HYBRID)
    h.step(next(_run()), _tools())
    assert _files(disk) == []
