"""Tests for the ``view_image`` tool's question parameter and description cache."""

from __future__ import annotations

import io
import os
from pathlib import Path

from langchain_core.messages import AIMessage
from PIL import Image

from backend.file_tools import _load_desc_cache, build_file_tools


def _png_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), os.urandom(w * h * 3)).save(buf, "PNG")
    return buf.getvalue()


class _FakeVLM:
    """Minimal stand-in for a vision LLM: records the prompts it receives."""

    model = "fake-vlm"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, messages):
        content = messages[0].content
        texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        self.prompts.append(texts[0] if texts else "")
        return AIMessage(content="A bar chart with three bars.")


def _get_view_image(files_dir: Path, vlm):
    tools = build_file_tools(files_dir, vision_llm=vlm)
    view_image = next(t for t in tools if t.name == "view_image")
    return view_image


def _write_upload(files_dir: Path) -> str:
    uploads = files_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    (uploads / "chart.png").write_bytes(_png_bytes())
    return "/uploads/chart.png"


def test_view_image_generic_description_is_cached(tmp_path: Path):
    rel = _write_upload(tmp_path)
    vlm = _FakeVLM()
    view_image = _get_view_image(tmp_path, vlm)

    out1 = view_image.invoke({"file_path": rel})
    assert "bar chart" in out1[0]["text"]
    assert len(vlm.prompts) == 1

    # Second generic view hits the cache — the VLM is not called again.
    out2 = view_image.invoke({"file_path": rel})
    assert "bar chart" in out2[0]["text"]
    assert len(vlm.prompts) == 1

    cache = _load_desc_cache(tmp_path)
    assert len(cache) == 1


def test_view_image_question_uses_prompt_and_is_not_cached(tmp_path: Path):
    rel = _write_upload(tmp_path)
    vlm = _FakeVLM()
    view_image = _get_view_image(tmp_path, vlm)

    question = "What is the value of the third bar?"
    out = view_image.invoke({"file_path": rel, "question": question})

    # The question is forwarded as the VLM prompt and echoed in the header.
    assert vlm.prompts == [question]
    assert question in out[0]["text"]

    # Targeted questions are never cached.
    assert _load_desc_cache(tmp_path) == {}


def test_view_image_question_not_cached_even_after_generic(tmp_path: Path):
    rel = _write_upload(tmp_path)
    vlm = _FakeVLM()
    view_image = _get_view_image(tmp_path, vlm)

    view_image.invoke({"file_path": rel})  # caches generic
    view_image.invoke({"file_path": rel, "question": "How many bars?"})

    # Generic cached (1 entry), question call re-invoked the VLM (2 prompts).
    assert len(vlm.prompts) == 2
    assert len(_load_desc_cache(tmp_path)) == 1
