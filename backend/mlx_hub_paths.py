"""Resolve Hugging Face Hub cache directory (HF_HUB_CACHE).

Paths **relative** to ``~/.cache`` (no leading ``/``, not starting with ``~``)
are joined under ``Path.home() / ".cache"``. Examples:

- ``""`` → ``~/.cache/huggingface/hub`` (same as Hugging Face default layout)
- ``"huggingface/hub"`` → ``~/.cache/huggingface/hub``
- ``"my-models"`` → ``~/.cache/my-models``
- ``"/Volumes/ssd/hf"`` or ``"~/Downloads/hf"`` → expanded absolute path
"""

from __future__ import annotations

from pathlib import Path


def resolve_hf_hub_cache_dir(raw: str | None) -> str:
    s = (raw or "").strip()
    if not s:
        return str((Path.home() / ".cache" / "huggingface" / "hub").resolve())
    if s.startswith("~"):
        return str(Path(s).expanduser().resolve())
    p = Path(s)
    if p.is_absolute():
        return str(p.expanduser().resolve())
    return str((Path.home() / ".cache" / p).resolve())


def default_hub_cache_relative_suffix() -> str:
    """Default folder segment under ``~/.cache`` (shown in Settings)."""
    return "huggingface/hub"
