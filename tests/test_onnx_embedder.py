"""The ONNX MiniLM path (used in the packaged app, which has no torch) must give the same vectors as
sentence-transformers, so stores built in dev stay valid in the bundle and the other way round.

Needs the model in the Hugging Face cache; skipped otherwise (CI without network)."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXTS = [
    "Mail: list emails",
    "Mail: Display all mailboxes and message viewer",
    "Numbers: export the open sheet to CSV",
    "",
    "a much longer sentence " * 80,  # past max_seq_length: truncated the same way
]


def _cached() -> bool:
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(MODEL, "onnx/model.onnx", local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _cached(), reason="MiniLM ONNX export not in the Hugging Face cache")


def test_onnx_matches_sentence_transformers():
    pytest.importorskip("torch")
    from sentence_transformers import SentenceTransformer

    from backend.embedding_index import _OnnxSentenceModel

    onnx = _OnnxSentenceModel.load(MODEL)
    ref = SentenceTransformer(MODEL).encode(TEXTS, normalize_embeddings=True).tolist()
    got = onnx.encode(TEXTS)
    assert onnx.dim == 384 and all(len(v) == 384 for v in got)
    for a, b in zip(got, ref):
        assert sum(x * y for x, y in zip(a, b)) > 0.9999


def test_embedder_works_without_torch_like_the_bundle():
    """scripts/build_backend.py excludes torch; the shared embedder must still load MiniLM."""
    code = (
        "import sys, asyncio\n"
        "for m in ('torch','torchvision','torchaudio','accelerate','datasets','sklearn'): sys.modules[m] = None\n"
        "from backend.embedding_index import _Embedder\n"
        f"e = _Embedder({MODEL!r})\n"
        "v = asyncio.run(e.embed(['Mail: list emails']))\n"
        "print(type(e._model).__name__, e.dim, len(v[0]))\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": f"{root}{os.pathsep}{root / 'src'}"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "_OnnxSentenceModel 384 384"


def test_procedural_memory_default_embedder_uses_onnx():
    from backend.procedural_memory.embed import SentenceEmbedder

    e = SentenceEmbedder(MODEL)
    v = asyncio.run(e.embed(["Mail: list emails"]))
    assert len(v[0]) == 384 and type(e._inner._model).__name__ == "_OnnxSentenceModel"
