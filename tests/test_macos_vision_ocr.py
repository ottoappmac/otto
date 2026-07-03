"""Tests for the macOS OCR read-and-act fallback tools.

These exercise the AX-disabled-app fallback (read_screen / find_text_on_screen
/ click_at ...) added for apps like Slack. They require the macOS pyobjc stack,
so the whole module is skipped elsewhere.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

if sys.platform != "darwin":  # pragma: no cover - platform guard
    pytest.skip("macОS-only (pyobjc)", allow_module_level=True)

try:
    from tools.navigation.computer.macos_tools import MacOSToolkit
except Exception as exc:  # pragma: no cover - missing frameworks
    pytest.skip(f"macos_tools unavailable: {exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def toolkit() -> MacOSToolkit:
    return MacOSToolkit()


def _tool(tk: MacOSToolkit, name: str):
    return next(t for t in tk.tools if t.name == name)


def test_ocr_tools_are_in_base_toolset(toolkit: MacOSToolkit):
    # OCR tools return text, so they must work for ANY model (not vision-gated).
    base = {t.name for t in toolkit.tools}
    for name in [
        "read_screen", "find_text_on_screen", "click_text",
        "click_at", "double_click_at", "right_click_at", "scroll_at",
    ]:
        assert name in base, f"{name} missing from base tools"

    # The pixel-screenshot tool stays vision-only (needs a VLM to consume it).
    vision = {t.name for t in toolkit.vision_tools}
    assert "capture_app_screenshot" in vision
    assert "read_screen" not in vision


def test_activate_and_verify_fails_for_missing_app(toolkit: MacOSToolkit):
    # A non-existent app can never become frontmost; verify it returns quickly
    # with success=False rather than falsely claiming activation.
    ok, _front = toolkit._activate_and_verify("NoSuchApp_zzz_123", timeout=0.6)
    assert ok is False


def test_type_text_focus_guard_refuses(toolkit: MacOSToolkit):
    # With app_name set and focus unobtainable, type_text must NOT type.
    out = asyncio.run(
        _tool(toolkit, "type_text").ainvoke(
            {"text": "hello", "app_name": "NoSuchApp_zzz_123"},
        )
    )
    assert "did not type" in out.lower()


def test_click_text_refuses_without_focus(toolkit: MacOSToolkit):
    out = asyncio.run(
        _tool(toolkit, "click_text").ainvoke(
            {"text": "Send", "app_name": "NoSuchApp_zzz_123"},
        )
    )
    assert "did not click" in out.lower()


def test_window_target_fullscreen(toolkit: MacOSToolkit):
    window_id, region = toolkit._window_target("")
    assert window_id is None
    x, y, w, h = region
    assert (x, y) == (0, 0)
    assert w > 0 and h > 0


def test_read_screen_missing_app(toolkit: MacOSToolkit):
    out = asyncio.run(
        _tool(toolkit, "read_screen").ainvoke(
            {"app_name": "NoSuchApp_zzz_123"},
        )
    )
    assert "no on-screen window" in out.lower()
    assert "activate_app" in out


def test_read_screen_text_formats_lines(toolkit: MacOSToolkit, monkeypatch):
    fake = [
        {"text": "Hello", "conf": 1.0, "x": 10, "y": 10},
        {"text": "world", "conf": 1.0, "x": 80, "y": 10},
        {"text": "second line", "conf": 1.0, "x": 10, "y": 200},
    ]
    monkeypatch.setattr(toolkit, "_window_target", lambda app: (1, (0, 0, 800, 600)))
    monkeypatch.setattr(
        toolkit, "_capture_for_read", lambda wid, region, want_image: (fake, None),
    )
    out = asyncio.run(_tool(toolkit, "read_screen").ainvoke({"app_name": "Slack"}))
    assert isinstance(out, str)
    assert "Hello world" in out
    assert "second line" in out


def test_read_screen_vision_returns_text_and_image(toolkit: MacOSToolkit, monkeypatch):
    # Vision variant must return a multimodal list: a text block + an image block,
    # built from the SAME capture (so coords/text/image agree).
    fake = [{"text": "Channel name", "conf": 1.0, "x": 10, "y": 10}]
    monkeypatch.setattr(toolkit, "_window_target", lambda app: (1, (0, 0, 800, 600)))
    monkeypatch.setattr(
        toolkit,
        "_capture_for_read",
        lambda wid, region, want_image: (fake, "QUJD" if want_image else None),
    )
    out = asyncio.run(toolkit.read_screen_vision.ainvoke({"app_name": "Slack"}))
    assert isinstance(out, list) and len(out) == 2
    text_block, image_block = out
    assert text_block["type"] == "text"
    assert "Channel name" in text_block["text"]
    assert image_block["type"] == "image_url"
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_block["image_url"]["url"].endswith("QUJD")


def test_find_text_ranking_prefers_whole_word(toolkit: MacOSToolkit, monkeypatch):
    # 'research' contains the substring 'search', but the real 'Search' box must
    # outrank it. Mock the capture so the test is deterministic and screen-free.
    fake = [
        {"text": "# research_and_dev", "conf": 1.0, "x": 10, "y": 400},
        {"text": "Search", "conf": 0.95, "x": 600, "y": 150},
    ]
    monkeypatch.setattr(toolkit, "_window_target", lambda app: (1, (0, 0, 800, 600)))
    monkeypatch.setattr(
        toolkit, "_capture_for_read", lambda wid, region, want_image: (fake, None),
    )

    out = asyncio.run(
        _tool(toolkit, "find_text_on_screen").ainvoke(
            {"text": "Search", "app_name": "Slack"},
        )
    )
    lines = [ln for ln in out.splitlines() if " at (" in ln]
    assert lines, out
    # Whole-word 'Search' must be listed before the substring match.
    assert lines[0].startswith("'Search'"), out


def test_find_text_no_match(toolkit: MacOSToolkit, monkeypatch):
    monkeypatch.setattr(toolkit, "_window_target", lambda app: (1, (0, 0, 800, 600)))
    monkeypatch.setattr(
        toolkit, "_capture_for_read",
        lambda wid, region, want_image: (
            [{"text": "Hello", "conf": 1.0, "x": 1, "y": 1}], None,
        ),
    )
    out = asyncio.run(
        _tool(toolkit, "find_text_on_screen").ainvoke(
            {"text": "zzz_nope", "app_name": "Slack"},
        )
    )
    assert "no on-screen text matching" in out.lower()


def test_read_screen_pruned_to_latest():
    # read_screen returns images for vision models; older results must be
    # superseded so screenshots don't pile up in context (Playwright-style).
    from langchain_core.messages import AIMessage, ToolMessage

    from agents.computer_voyager import (
        _SCREEN_CONTROL_TOOLS,
        _SUPERSEDED_PLACEHOLDER,
        _prune_screen_controls,
    )

    assert "read_screen" in _SCREEN_CONTROL_TOOLS

    messages = [
        AIMessage(content="", tool_calls=[{"name": "read_screen", "args": {}, "id": "1"}]),
        ToolMessage(content="OLD screen text", name="read_screen", tool_call_id="1"),
        AIMessage(content="", tool_calls=[{"name": "read_screen", "args": {}, "id": "2"}]),
        ToolMessage(content="NEW screen text", name="read_screen", tool_call_id="2"),
    ]
    pruned = _prune_screen_controls(messages)
    assert pruned[1].content == _SUPERSEDED_PLACEHOLDER
    assert pruned[3].content == "NEW screen text"
