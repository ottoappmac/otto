"""Curated STT + wake-word model catalog with hardware-aware fit scoring.

Mirrors the structure of :mod:`backend.mlx_catalog` but for speech models.

Two layers (identical pattern to mlx_catalog):

1. ``VOICE_CURATED`` — a static list of :class:`VoiceCatalogRow` entries.
2. :func:`fetch_voice_catalog` — returns ``VOICE_CURATED`` augmented with
   live HF download counts, cached on disk for 24 h.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog row shape
# ---------------------------------------------------------------------------


@dataclass
class VoiceCatalogRow:
    """One curated speech model.

    ``weights_gb`` is the measured on-disk safetensor / ONNX size.
    ``kind`` drives fit scoring (different fixed overhead per kind).
    """

    repo_id: str
    kind: str           # "stt" | "wake"
    display_name: str
    blurb: str
    weights_gb: float
    format: str         # "mlx" | "onnx"
    language: str       # "multilingual" | "en" | "en,es,fr,..."
    latency_class: str  # "realtime" | "near-realtime" | "batch"
    wake_phrases: list[str] = field(default_factory=list)
    featured: bool = False
    # Filled by fetch_voice_catalog() from HF Hub
    downloads: int = 0
    last_modified: str = ""


# ---------------------------------------------------------------------------
# Curated list
# ---------------------------------------------------------------------------

VOICE_CURATED: list[VoiceCatalogRow] = [

    # ── Wake word: bundled model ──────────────────────────────────────────
    VoiceCatalogRow(
        repo_id="__builtin__/hey_otto",
        kind="wake",
        display_name="Hey Otto",
        blurb="Built-in wake word — say 'Hey Otto' to activate hands-free. No download needed.",
        weights_gb=0.002,
        format="onnx",
        language="en",
        latency_class="realtime",
        wake_phrases=["Hey Otto"],
        featured=True,
    ),

    # ── STT: Whisper MLX ports ────────────────────────────────────────────
    VoiceCatalogRow(
        repo_id="mlx-community/whisper-large-v3-turbo",
        kind="stt",
        display_name="Whisper Large v3 Turbo",
        blurb="~809 MB · 99-lang · best accuracy, near-realtime on Apple Silicon. Recommended.",
        weights_gb=0.79,
        format="mlx",
        language="multilingual",
        latency_class="near-realtime",
        featured=True,
    ),
    VoiceCatalogRow(
        repo_id="mlx-community/whisper-small.en-mlx",
        kind="stt",
        display_name="Whisper Small (English)",
        blurb="~242 MB · English only · fast transcription, great for low-power Macs.",
        weights_gb=0.24,
        format="mlx",
        language="en",
        latency_class="realtime",
        featured=True,
    ),
    VoiceCatalogRow(
        repo_id="mlx-community/whisper-base.en",
        kind="stt",
        display_name="Whisper Base (English)",
        blurb="~74 MB · English only · lowest latency, suitable for all Macs.",
        weights_gb=0.07,
        format="mlx",
        language="en",
        latency_class="realtime",
    ),
    VoiceCatalogRow(
        repo_id="mlx-community/whisper-medium-mlx",
        kind="stt",
        display_name="Whisper Medium",
        blurb="~769 MB · multilingual · good balance of accuracy and speed.",
        weights_gb=0.75,
        format="mlx",
        language="multilingual",
        latency_class="near-realtime",
    ),

]


# ---------------------------------------------------------------------------
# Fit scoring
# ---------------------------------------------------------------------------

_OVERHEAD_GB: dict[str, float] = {
    "stt": 0.3,   # encoder + decoder runtime buffers
    "wake": 0.0,  # tiny CPU-only ONNX; essentially free
}


def score_voice_row(
    row: VoiceCatalogRow,
    *,
    ram_gb: float,
    wired_limit_gb: float,
    free_disk_gb: float,
    comfortable_fraction: float = 0.5,
) -> dict[str, Any]:
    """Score ``row`` against probed hardware."""
    overhead = _OVERHEAD_GB.get(row.kind, 0.3)
    total = round(row.weights_gb + overhead, 2)

    if row.kind == "wake":
        fits = "comfortable"
    elif ram_gb <= 0:
        fits = "unknown"
    else:
        comfortable_budget = comfortable_fraction * ram_gb
        ceiling = max(comfortable_budget, wired_limit_gb)
        if total <= comfortable_budget:
            fits = "comfortable"
        elif total <= ceiling:
            fits = "tight"
        else:
            fits = "over"

    disk_ok = free_disk_gb <= 0 or (row.weights_gb + 0.2) <= free_disk_gb

    return {
        "weights_gb": round(row.weights_gb, 2),
        "overhead_gb": round(overhead, 2),
        "total_gb": total,
        "fits": fits,
        "disk_ok": disk_ok,
    }


def score_voice_catalog(
    rows: Iterable[VoiceCatalogRow],
    *,
    ram_gb: float,
    wired_limit_gb: float,
    free_disk_gb: float,
    comfortable_fraction: float = 0.5,
    cached_map: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Score every row and sort comfortable → tight → over."""
    cached: dict[str, bool] = cached_map or {}
    out: list[dict[str, Any]] = []
    for r in rows:
        score = score_voice_row(
            r,
            ram_gb=ram_gb,
            wired_limit_gb=wired_limit_gb,
            free_disk_gb=free_disk_gb,
            comfortable_fraction=comfortable_fraction,
        )
        out.append({
            **asdict(r),
            **score,
            "already_cached": r.repo_id in cached or r.repo_id.startswith("__builtin__"),
            "cache_complete": bool(cached.get(r.repo_id, False)) or r.repo_id.startswith("__builtin__"),
        })

    fit_rank = {"comfortable": 0, "tight": 1, "unknown": 2, "over": 3}
    kind_rank = {"wake": 0, "stt": 1}
    out.sort(
        key=lambda x: (
            kind_rank.get(str(x.get("kind", "")), 9),
            fit_rank.get(str(x.get("fits", "")), 9),
            0 if x.get("featured") else 1,
            -int(x.get("downloads", 0) or 0),
            x.get("display_name", ""),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Live enrichment from HF Hub (24-hour disk cache)
# ---------------------------------------------------------------------------

_SPEECH_RE = re.compile(
    r"(?:\b|_)(?:whisper|parakeet|moonshine|sensevoice|sense[\s_-]?voice)(?:\b|_)",
    re.I,
)

_VOICE_CACHE_FILENAME = "voice_catalog_cache.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60

_enrichment_task: asyncio.Task | None = None  # type: ignore[type-arg]


def _cache_path() -> Path:
    from backend.config import get_app_data_dir
    return get_app_data_dir() / _VOICE_CACHE_FILENAME


def _load_cache() -> dict[str, dict[str, Any]] | None:
    path = _cache_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if time.time() - float(data.get("fetched_at", 0)) > _CACHE_TTL_SECONDS:
        return None
    rows = data.get("rows")
    return rows if isinstance(rows, dict) else None


def _save_cache(rows: dict[str, dict[str, Any]]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": time.time(), "rows": rows}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("could not write voice catalog cache: %s", exc)


def _enrich_blocking(
    repo_ids: list[str],
    token: str | None,
    *,
    dynamic_limit: int = 200,
) -> dict[str, dict[str, Any]]:
    """Fetch download counts for curated repos and discover community STT models."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {}
    api = HfApi(token=token) if token else HfApi()
    out: dict[str, dict[str, Any]] = {}
    wanted = set(repo_ids)
    limit = max(len(repo_ids) + 20, dynamic_limit)
    try:
        listing = api.list_models(
            author="mlx-community",
            sort="downloads",
            limit=limit,
            expand=["downloads", "lastModified"],
        )
        for info in listing:
            rid = getattr(info, "id", "") or getattr(info, "modelId", "")
            if not rid:
                continue
            slug = rid.split("/")[-1].lower()
            if not (_SPEECH_RE.search(slug) or rid in wanted):
                continue
            downloads = int(getattr(info, "downloads", 0) or 0)
            lm_raw = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
            lm = ""
            try:
                lm = lm_raw.isoformat() if lm_raw and hasattr(lm_raw, "isoformat") else str(lm_raw or "")
            except Exception:  # noqa: BLE001
                pass
            out[rid] = {"downloads": downloads, "last_modified": lm}
    except Exception as exc:  # noqa: BLE001
        logger.debug("voice catalog HF list_models failed: %s", exc)
    return out


async def _run_enrichment(repo_ids: list[str], token: str | None) -> None:
    global _enrichment_task  # noqa: PLW0603
    try:
        live = await asyncio.to_thread(_enrich_blocking, repo_ids, token)
        if live:
            _save_cache(live)
    except Exception as exc:  # noqa: BLE001
        logger.debug("voice catalog enrichment failed: %s", exc)


def _start_enrichment(repo_ids: list[str], token: str | None) -> None:
    global _enrichment_task  # noqa: PLW0603
    if _enrichment_task is not None and not _enrichment_task.done():
        return
    try:
        loop = asyncio.get_running_loop()
        _enrichment_task = loop.create_task(
            _run_enrichment(repo_ids, token),
            name="voice_catalog_enrichment",
        )
    except RuntimeError:
        pass


def is_enriching() -> bool:
    return _enrichment_task is not None and not _enrichment_task.done()


async def fetch_voice_catalog(*, token: str | None = None) -> list[VoiceCatalogRow]:
    """Return curated rows merged with dynamically discovered STT models."""
    curated_ids = {r.repo_id for r in VOICE_CURATED}
    rows = [VoiceCatalogRow(**asdict(r)) for r in VOICE_CURATED]
    hf_ids = [r.repo_id for r in rows if not r.repo_id.startswith("__builtin__")]

    cached = _load_cache()
    if cached is None:
        live: dict[str, dict[str, Any]] = {}
        _start_enrichment(hf_ids, token)
    else:
        live = cached

    for r in rows:
        meta = live.get(r.repo_id, {})
        r.downloads = int(meta.get("downloads", 0) or 0)
        r.last_modified = str(meta.get("last_modified", ""))

    for rid, meta in live.items():
        if rid in curated_ids:
            continue
        slug = rid.split("/")[-1].lower()
        if not _SPEECH_RE.search(slug):
            continue
        rows.append(VoiceCatalogRow(
            repo_id=rid,
            kind="stt",
            display_name=rid.split("/")[-1].replace("-", " ").replace("_", " "),
            blurb="",
            weights_gb=0.0,
            format="mlx",
            language="multilingual",
            latency_class="near-realtime",
            downloads=int(meta.get("downloads", 0) or 0),
            last_modified=str(meta.get("last_modified", "")),
        ))

    return rows


__all__ = [
    "VoiceCatalogRow",
    "VOICE_CURATED",
    "score_voice_row",
    "score_voice_catalog",
    "fetch_voice_catalog",
    "is_enriching",
]
