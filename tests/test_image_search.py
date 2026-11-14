"""image_search: saves reference images from a web search for view_image."""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from backend import file_tools
from backend.agent_library import _CAD_AGENT_PROMPT


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "PNG")
    return buf.getvalue()


class _FakeDDGS:
    def __init__(self, hits):
        self.hits = hits

    def images(self, query, **kw):
        return self.hits


def _patch(monkeypatch, hits, fetch):
    import tools.research._loaders as loaders

    monkeypatch.setattr(loaders, "_get_ddgs", lambda: _FakeDDGS(hits))
    monkeypatch.setattr(file_tools, "_fetch_image_bytes", fetch)


def test_saves_good_images_and_skips_failures(monkeypatch, tmp_path: Path):
    hits = [{"image": f"https://x/{i}.png", "title": f"t{i}", "url": f"https://p/{i}"} for i in range(5)]

    def fetch(url):
        return "Error: HTTP 403 fetching URL." if url.endswith("0.png") else (_png(), "image/png")

    _patch(monkeypatch, hits, fetch)
    out = file_tools._search_and_save_images("chess knight", tmp_path / "ref", 2)
    files = sorted((tmp_path / "ref").iterdir())
    assert len(files) == 2 and all(f.suffix == ".png" for f in files)
    assert "view_image" in out and str(files[0]) in out


def test_no_hits_and_no_downloads(monkeypatch, tmp_path: Path):
    _patch(monkeypatch, [], lambda u: (_png(), "image/png"))
    assert file_tools._search_and_save_images("q", tmp_path / "r", 3).startswith("No images found")
    _patch(monkeypatch, [{"image": "https://x/a.png"}], lambda u: "Error: nope")
    assert "none could be downloaded" in file_tools._search_and_save_images("q", tmp_path / "r", 3)


def test_fetch_rejects_private_and_bad_scheme():
    assert file_tools._fetch_image_bytes("ftp://x/a.png").startswith("Error")
    assert "private" in file_tools._fetch_image_bytes("http://127.0.0.1/a.png")


def test_tool_registered_and_prompt_mentions_it(tmp_path: Path):
    names = {t.name for t in file_tools.build_file_tools(tmp_path)}
    assert {"view_image", "load_image_from_url", "image_search"} <= names
    assert "image_search" in _CAD_AGENT_PROMPT
