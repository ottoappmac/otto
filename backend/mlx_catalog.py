"""Curated MLX model catalog + hardware-aware fit scoring.

The On-Device tab's "no models cached yet" empty state is the
biggest first-run friction point — users have to invent a Hugging
Face repo id from memory and hope it works on their machine.  This
module replaces that with a small, hand-picked list of MLX models
where every row knows its weight footprint and architecture
parameters, so we can score each one against the user's actual RAM
+ disk and label it green / amber / red.

Two layers:

1. ``CURATED`` — a static list of :class:`CatalogRow` entries, edited
   by hand.  Architecture fields (``n_layers``, ``n_kv_heads``,
   ``head_dim``) are filled from each model's published ``config.json``
   so the KV-cache estimate doesn't need any network round-trip.
2. :func:`fetch_catalog` — returns ``CURATED`` augmented (when network
   permits) with live download counts via a single
   :func:`huggingface_hub.HfApi.list_models` call, cached on disk for
   24 h.  All scoring is done by :func:`score_catalog` against the
   probe in :mod:`backend.setup_capabilities`.

Estimates are intentionally bucketed (comfortable / tight / over) —
they're good to ~10%, not exact, and the UI never shows raw GB as if
it were truth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
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
class CatalogRow:
    """One curated MLX-community model.

    All fields are static / hand-curated except ``downloads`` and
    ``last_modified`` which are filled by :func:`fetch_catalog` from
    the live Hub listing when available.

    Architecture fields drive the KV-cache estimate:

    ``kv_cache_bytes ≈ 2 · n_layers · n_kv_heads · head_dim · ctx · kv_dtype_bytes``
    """

    repo_id: str
    family: str                # "qwen3" | "llama3" | "gemma3" | "phi4" | "mistral" | "deepseek-r1" | "qwen-vlm"
    display_name: str
    blurb: str                 # one-line "what this is good for"
    weights_gb: float          # measured bytes-on-disk for the safetensors shards
    params_b: float            # billions of parameters (for display)
    quant: str                 # "4bit" | "8bit" | "bf16" | "2bit" | "mixed"
    role: list[str]            # ["text"] | ["vlm"] | ["draft"] | ["text","tools"]
    capability_tags: list[str] = field(default_factory=list)
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    max_position: int = 32768
    requires_token: bool = False
    tier: str = "balanced"     # "starter" | "balanced" | "power"
    featured: bool = False
    # Filled by fetch_catalog() — never serialised into the curated source.
    downloads: int = 0
    last_modified: str = ""


# ---------------------------------------------------------------------------
# Curated list — edit by hand
# ---------------------------------------------------------------------------
#
# Picking criteria:
#   * Available under ``mlx-community/``.
#   * Tool-calling supported by ``src/chat_models/mlx/_native_tool_parsing.py``
#     for everything tagged "tools" (Qwen / Llama / Mistral / Phi families).
#   * Quantisation that makes sense on a Mac (4-bit dominates; a couple
#     of 8-bit options for headroom users).
#   * Three rough tiers — starter (≤4 GB), balanced (4–10 GB), power (≥10 GB).


CURATED: list[CatalogRow] = [
    # ─── Starter (≤ 4 GB on disk) ──────────────────────────────────────
    CatalogRow(
        repo_id="mlx-community/Qwen3-1.7B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 1.7B (4-bit)",
        blurb="Tiny, fast, runs on 8 GB Macs. Surprisingly capable for tools.",
        weights_gb=1.1,
        params_b=1.7,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking"],
        n_layers=28, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="starter", featured=True,
    ),
    CatalogRow(
        repo_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
        family="llama3",
        display_name="Llama 3.2 — 3B Instruct (4-bit)",
        blurb="Solid general-purpose chat model under 2 GB.",
        weights_gb=1.8,
        params_b=3.2,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools"],
        n_layers=28, n_kv_heads=8, head_dim=128, max_position=131072,
        tier="starter",
    ),
    CatalogRow(
        repo_id="mlx-community/Phi-4-mini-instruct-4bit",
        family="phi4",
        display_name="Phi-4 mini — Instruct (4-bit)",
        blurb="Microsoft's compact reasoning model. Strong at code + math.",
        weights_gb=2.3,
        params_b=3.8,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "reasoning", "code"],
        n_layers=32, n_kv_heads=8, head_dim=96, max_position=131072,
        tier="starter",
    ),

    CatalogRow(
        repo_id="mlx-community/Qwen3-4B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 4B (4-bit)",
        blurb="Sweet spot between the 1.7B and 8B. Tool-calling, fast on 8 GB Macs.",
        weights_gb=2.3,
        params_b=4.0,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking"],
        n_layers=36, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="starter", featured=True,
    ),

    # ─── Balanced (4–10 GB) ────────────────────────────────────────────
    CatalogRow(
        repo_id="mlx-community/Qwen3-8B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 8B (4-bit)",
        blurb="Daily-driver chat model. Tool-calling, 32K context, fits 16 GB Macs.",
        weights_gb=4.6,
        params_b=8.2,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking"],
        n_layers=36, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="balanced", featured=True,
    ),
    CatalogRow(
        repo_id="mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        family="mistral",
        display_name="Mistral Nemo — 12B (4-bit)",
        blurb="Mistral + NVIDIA collab. 128K context, very fluent prose.",
        weights_gb=6.9,
        params_b=12.2,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "long-context"],
        n_layers=40, n_kv_heads=8, head_dim=128, max_position=131072,
        tier="balanced",
    ),
    CatalogRow(
        repo_id="mlx-community/gemma-3-12b-it-4bit",
        family="gemma3",
        display_name="Gemma 3 — 12B IT (4-bit)",
        blurb="Google's instruction-tuned model. Multilingual, 128K context.",
        weights_gb=7.4,
        params_b=12.0,
        quant="4bit",
        role=["text"],
        capability_tags=["chat", "long-context", "multilingual"],
        n_layers=48, n_kv_heads=4, head_dim=256, max_position=131072,
        tier="balanced",
    ),
    CatalogRow(
        repo_id="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        family="qwen-vlm",
        display_name="Qwen 2.5 VL — 7B Instruct (4-bit)",
        blurb="Vision + text. Good for screenshots, charts, document QA.",
        weights_gb=5.1,
        params_b=8.3,
        quant="4bit",
        role=["vlm"],
        capability_tags=["vision", "chat"],
        n_layers=28, n_kv_heads=4, head_dim=128, max_position=32768,
        tier="balanced", featured=True,
    ),
    CatalogRow(
        repo_id="mlx-community/Qwen3-14B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 14B (4-bit)",
        blurb="Strong reasoning and coding. Fits comfortably on 32 GB Macs.",
        weights_gb=8.2,
        params_b=14.7,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking", "reasoning"],
        n_layers=40, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="balanced", featured=True,
    ),
    CatalogRow(
        repo_id="mlx-community/DeepSeek-R1-Distill-Qwen-14B-8bit",
        family="deepseek-r1",
        display_name="DeepSeek R1 Distill — Qwen 14B (8-bit)",
        blurb="Distilled reasoning model. Stronger thinking traces, slower.",
        weights_gb=15.7,
        params_b=14.8,
        quant="8bit",
        role=["text"],
        capability_tags=["chat", "reasoning", "thinking"],
        n_layers=48, n_kv_heads=8, head_dim=128, max_position=131072,
        tier="balanced",
    ),

    # ─── Power (≥10 GB; for 32 GB+ Macs) ───────────────────────────────
    CatalogRow(
        repo_id="mlx-community/Qwen3-30B-A3B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 30B-A3B MoE (4-bit)",
        blurb="Mixture-of-Experts: 30B total weights, 3B active. Dense-model speed with big-model smarts.",
        weights_gb=15.5,
        params_b=30.0,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking"],
        n_layers=48, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="power",
    ),
    CatalogRow(
        repo_id="mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
        family="qwen3",
        display_name="Qwen 3.6 — 35B-A3B MTP (4-bit)",
        blurb="Newest Qwen MoE with Multi-Token Prediction. 35B weights, 3B active — fast and capable.",
        weights_gb=17.1,
        params_b=35.0,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking", "reasoning"],
        n_layers=0, n_kv_heads=0, head_dim=0, max_position=32768,
        tier="power", featured=True,
    ),
    CatalogRow(
        repo_id="mlx-community/Qwen3.6-35B-A3B-MTP-5bit",
        family="qwen3",
        display_name="Qwen 3.6 — 35B-A3B MTP (5-bit)",
        blurb="Higher-precision variant of the Qwen 3.6 MoE. Slightly better quality, ~25% more RAM.",
        weights_gb=21.4,
        params_b=35.0,
        quant="5bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking", "reasoning"],
        n_layers=0, n_kv_heads=0, head_dim=0, max_position=32768,
        tier="power",
    ),
    CatalogRow(
        repo_id="mlx-community/Qwen3-32B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 32B (4-bit)",
        blurb="Big-Mac flagship. Excellent reasoning, tool-calling, long context.",
        weights_gb=18.2,
        params_b=32.5,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "thinking", "reasoning"],
        n_layers=64, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="power", featured=True,
    ),
    CatalogRow(
        repo_id="mlx-community/Llama-3.3-70B-Instruct-4bit",
        family="llama3",
        display_name="Llama 3.3 — 70B Instruct (4-bit)",
        blurb="Frontier-quality open weights. Needs 64 GB+ unified memory.",
        weights_gb=39.7,
        params_b=70.6,
        quant="4bit",
        role=["text", "tools"],
        capability_tags=["chat", "tools", "long-context"],
        n_layers=80, n_kv_heads=8, head_dim=128, max_position=131072,
        tier="power",
    ),

    # ─── Speculative-decoding draft pairs ──────────────────────────────
    CatalogRow(
        repo_id="mlx-community/Qwen3-0.6B-4bit",
        family="qwen3",
        display_name="Qwen 3 — 0.6B (4-bit, draft)",
        blurb="Speculative-decoding draft for any Qwen 3 variant.",
        weights_gb=0.4,
        params_b=0.6,
        quant="4bit",
        role=["draft"],
        capability_tags=["draft"],
        n_layers=28, n_kv_heads=8, head_dim=128, max_position=32768,
        tier="starter",
    ),
]


# ---------------------------------------------------------------------------
# Dynamic model discovery helpers
# ---------------------------------------------------------------------------
#
# These functions infer a CatalogRow from a bare HuggingFace repo id when
# the model is not in CURATED.  Metadata accuracy is lower than hand-curated
# rows (architecture fields default to 0, which falls back to reasonable
# defaults inside estimate_footprint_gb), but it's enough to show the model
# in the picker with a fit badge and download button.

_FAMILY_HINTS: list[tuple[str, str]] = [
    (r"qwen3\.6|qwen3.6", "qwen3.6"),
    (r"qwen3", "qwen3"),
    (r"qwen[\s_-]?2\.?5[\s_-]?vl|qwen.*[\s_-]vl\b", "qwen-vlm"),
    (r"qwen2\.?5|qwen2|qwen", "qwen2"),
    (r"llama[\s_-]?3", "llama3"),
    (r"llama[\s_-]?2", "llama2"),
    (r"mistral", "mistral"),
    (r"gemma[\s_-]?3|gemma[\s_-]?2|gemma", "gemma3"),
    (r"phi[\s_-]?4|phi[\s_-]?3\.?5|phi[\s_-]?3|phi", "phi4"),
    (r"deepseek[\s_-]?r1", "deepseek-r1"),
    (r"deepseek", "deepseek"),
    (r"smollm|smol[\s_-]?lm", "smollm"),
    (r"falcon", "falcon"),
    (r"internvl|intern[\s_-]?vl", "internvl"),
    (r"command[\s_-]?r", "cohere"),
]

# Repos that match these patterns are very likely not chat/LLMs (embedders,
# rerankers, image models, and speech ASR/TTS models that would be useless
# in a chat-model picker).
_SKIP_RE = re.compile(
    r"(?:\b|_)(?:"
    r"adapter|lora|gguf|"
    # Embedding / reranking / encoders
    r"bert|roberta|bge|all[\s_-]?minilm|minilm|sentence[\s_-]?transformer|"
    r"e5|nomic|embed|embedding|reranker|reward|gte[\s_-]|jina[\s_-]|"
    # Image / diffusion / vision-only encoders
    r"stable[\s_-]?diffusion|sdxl|flux|clip|siglip|mxbai|vae|controlnet|"
    # Speech: ASR / TTS / audio
    r"whisper|parakeet|kokoro|tts|tdt|ctc|moonshine|sense[\s_-]?voice|"
    r"sensevoice|encodec|musicgen|bark|xtts|f5[\s_-]?tts|outetts|"
    r"csm[\s_-]|dia[\s_-]|vad|speech|audio|wav2vec|"
    r"orca[\s_-]?mini"
    r")(?:\b|_)",
    re.I,
)


def _infer_family(slug_lower: str) -> str:
    for pattern, family in _FAMILY_HINTS:
        if re.search(pattern, slug_lower):
            return family
    return "other"


def _infer_params_b(slug: str) -> float:
    """Parse parameter count from a repo slug like 'Qwen3-8B-4bit' → 8.0."""
    m = re.search(r"(\d+\.?\d*)\s*[Bb](?:\b|[\s_\-]|$)", slug)
    if m:
        v = float(m.group(1))
        if 0 < v <= 500:
            return v
    return 0.0


def _infer_quant(slug_lower: str) -> str:
    m = re.search(r"(\d+)[\s_-]?bit", slug_lower)
    if m:
        return f"{m.group(1)}bit"
    for marker in ("bf16", "fp16", "fp32"):
        if marker in slug_lower:
            return marker
    return "4bit"


def _estimate_weights_gb(params_b: float, quant: str) -> float:
    if params_b <= 0:
        return 0.0
    bpp: dict[str, float] = {
        "2bit": 2, "3bit": 3, "4bit": 4, "5bit": 5, "6bit": 6, "8bit": 8,
        "bf16": 16, "fp16": 16, "fp32": 32,
    }
    return round(params_b * 1e9 * bpp.get(quant, 4) / 8 / (1024 ** 3) * 1.05, 1)


def _infer_role_tags(slug: str) -> tuple[list[str], list[str]]:
    lower = slug.lower()
    if re.search(r"\bvl\b|vision|llava|paligemma|moondream|internvl", lower):
        return ["vlm"], ["vision", "chat"]
    if re.search(r"\b(0\.6b|draft|speculative)\b", lower):
        return ["draft"], ["draft"]
    tags: list[str] = ["chat"]
    if any(fam in lower for fam in ["qwen", "llama", "mistral", "phi", "command-r"]):
        tags.append("tools")
    if re.search(r"\b(think|reason|r1|qwq|cot)\b", lower):
        tags.append("thinking")
    role = ["text", "tools"] if "tools" in tags else ["text"]
    return role, tags


def _make_display_name(slug: str) -> str:
    """Turn 'Qwen3-8B-Instruct-4bit' into a readable label."""
    name = re.sub(r"[-_](\d+)bit$", r" (\1-bit)", slug, flags=re.I)
    name = re.sub(r"[-_](bf16|fp16|fp32)$", r" (\1)", name, flags=re.I)
    name = re.sub(r"[-_]([Ii]nstruct|IT)\b", " Instruct", name, flags=re.I)
    name = re.sub(r"[-_]([Cc]hat)\b", " Chat", name, flags=re.I)
    name = re.sub(r"[-_](VL)\b", " VL", name)
    name = name.replace("-", " ").replace("_", " ")
    return name


def _infer_row_from_hub(
    repo_id: str,
    *,
    downloads: int = 0,
    last_modified: str = "",
) -> CatalogRow | None:
    """Synthesise a CatalogRow from a HuggingFace repo id.

    Returns ``None`` for repos that look like non-LLMs (adapters, embedders,
    ASR models, …) or where we cannot estimate a meaningful footprint.
    """
    slug = repo_id.split("/")[-1]
    lower = slug.lower()

    if _SKIP_RE.search(lower):
        return None

    params_b = _infer_params_b(slug)
    if params_b <= 0:
        return None  # Can't estimate footprint — skip.

    quant = _infer_quant(lower)
    weights_gb = _estimate_weights_gb(params_b, quant)
    family = _infer_family(lower)
    role, tags = _infer_role_tags(slug)
    tier = "starter" if weights_gb <= 4 else "balanced" if weights_gb <= 12 else "power"

    return CatalogRow(
        repo_id=repo_id,
        family=family,
        display_name=_make_display_name(slug),
        blurb="",
        weights_gb=weights_gb,
        params_b=params_b,
        quant=quant,
        role=role,
        capability_tags=tags,
        n_layers=0,
        n_kv_heads=0,
        head_dim=0,
        tier=tier,
        downloads=downloads,
        last_modified=last_modified,
    )


# ---------------------------------------------------------------------------
# Footprint + fit scoring
# ---------------------------------------------------------------------------


# Per-token KV bytes per layer per head equals 2 (K and V) * head_dim *
# kv_dtype_bytes.  This constant is multiplied by ``n_layers *
# n_kv_heads * head_dim * ctx_len * kv_dtype_bytes`` below.
_FRAMEWORK_OVERHEAD_GB = 0.5
_ACTIVATION_OVERHEAD_GB = 1.0
_VLM_EXTRA_GB = 1.5  # vision encoder + image tensors


def _kv_dtype_bytes(kv_bits: int | None, weights_quant: str) -> float:
    """How many bytes per KV value at runtime."""
    if kv_bits in (4,):
        return 0.5
    if kv_bits in (8,):
        return 1.0
    # Default unquantised KV is fp16 / bf16 regardless of weight precision.
    return 2.0


def estimate_footprint_gb(
    row: CatalogRow,
    *,
    ctx_len: int,
    kv_bits: int | None,
    include_draft_gb: float = 0.0,
) -> dict[str, float]:
    """Estimate runtime memory for ``row`` at ``ctx_len`` tokens.

    Returns a breakdown dict so the UI can show a tooltip.  Values are
    deliberately rounded to one decimal place — they're not precise
    enough to justify more.
    """
    kv_bpv = _kv_dtype_bytes(kv_bits, row.quant)
    layers = max(1, row.n_layers or 32)
    heads = max(1, row.n_kv_heads or 8)
    hdim = max(1, row.head_dim or 128)
    ctx = max(512, int(ctx_len))

    kv_bytes = 2.0 * layers * heads * hdim * ctx * kv_bpv
    kv_gb = kv_bytes / (1024 ** 3)

    overhead_gb = _FRAMEWORK_OVERHEAD_GB + _ACTIVATION_OVERHEAD_GB
    if "vlm" in row.role:
        overhead_gb += _VLM_EXTRA_GB

    total = row.weights_gb + kv_gb + overhead_gb + include_draft_gb

    return {
        "weights_gb": round(row.weights_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "draft_gb": round(include_draft_gb, 2),
        "total_gb": round(total, 2),
    }


def score_row(
    row: CatalogRow,
    *,
    ram_gb: float,
    wired_limit_gb: float,
    free_disk_gb: float,
    ctx_len: int,
    kv_bits: int | None,
    comfortable_fraction: float = 0.5,
) -> dict[str, Any]:
    """Score ``row`` against a probed machine.

    Returns the breakdown plus a ``fits`` enum:
    ``comfortable`` (≤ ``comfortable_fraction`` × RAM),
    ``tight`` (≤ wired-limit ceiling) or
    ``over`` (would not load).
    Also flags ``disk_ok`` separately so the UI can warn even when
    the model would fit in RAM.
    """
    breakdown = estimate_footprint_gb(row, ctx_len=ctx_len, kv_bits=kv_bits)
    total = breakdown["total_gb"]

    comfortable_budget = max(0.0, comfortable_fraction * ram_gb)
    ceiling = max(comfortable_budget, wired_limit_gb)

    if ram_gb <= 0:
        fits = "unknown"
    elif total <= comfortable_budget:
        fits = "comfortable"
    elif total <= ceiling:
        fits = "tight"
    else:
        fits = "over"

    disk_ok = free_disk_gb <= 0 or row.weights_gb + 1.0 <= free_disk_gb

    return {
        **breakdown,
        "fits": fits,
        "headroom_gb": round(comfortable_budget - total, 2),
        "ceiling_gb": round(ceiling, 2),
        "disk_ok": disk_ok,
    }


def score_catalog(
    rows: Iterable[CatalogRow],
    *,
    ram_gb: float,
    wired_limit_gb: float,
    free_disk_gb: float,
    ctx_len: int = 8192,
    kv_bits: int | None = None,
    comfortable_fraction: float = 0.5,
    cached_map: dict[str, bool] | None = None,
    # Legacy name kept for call sites that haven't migrated yet.
    cached_repo_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Score every row, sort comfortable→tight→over, then by quality."""
    # ``cached_map`` is {repo_id: is_complete}; ``cached_repo_ids`` is the
    # old set-only form kept for backward compat.
    if cached_map is not None:
        cached: dict[str, bool] = cached_map
    elif cached_repo_ids is not None:
        cached = {rid: True for rid in cached_repo_ids}
    else:
        cached = {}

    out: list[dict[str, Any]] = []
    for r in rows:
        score = score_row(
            r,
            ram_gb=ram_gb,
            wired_limit_gb=wired_limit_gb,
            free_disk_gb=free_disk_gb,
            ctx_len=ctx_len,
            kv_bits=kv_bits,
            comfortable_fraction=comfortable_fraction,
        )
        out.append({
            **asdict(r),
            **score,
            "already_cached": r.repo_id in cached,
            "cache_complete": bool(cached.get(r.repo_id, False)),
        })

    fit_rank = {"comfortable": 0, "tight": 1, "unknown": 2, "over": 3}
    out.sort(
        key=lambda x: (
            fit_rank.get(str(x.get("fits", "")), 9),
            0 if x.get("featured") else 1,
            -int(x.get("downloads", 0) or 0),
            -float(x.get("params_b", 0) or 0),
            x.get("display_name", ""),
        )
    )
    return out


# ---------------------------------------------------------------------------
# Live enrichment from HF Hub (downloads, last_modified)
# ---------------------------------------------------------------------------


_CACHE_FILENAME = "mlx_catalog_cache.json"
_CACHE_TTL_SECONDS = 24 * 60 * 60


def _cache_path() -> Path:
    from backend.config import get_app_data_dir

    return get_app_data_dir() / _CACHE_FILENAME


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
    fetched_at = float(data.get("fetched_at", 0))
    if time.time() - fetched_at > _CACHE_TTL_SECONDS:
        return None
    rows = data.get("rows")
    if not isinstance(rows, dict):
        return None
    return rows


def _save_cache(rows: dict[str, dict[str, Any]]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"fetched_at": time.time(), "rows": rows}, indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.debug("could not write catalog cache: %s", exc)


def _enrich_blocking(
    repo_ids: list[str],
    token: str | None,
    *,
    dynamic_limit: int = 1000,
) -> dict[str, dict[str, Any]]:
    """Fetch download counts + last-modified for curated repos AND discover
    the top ``dynamic_limit`` mlx-community models by downloads.

    Previously this filtered the HF listing to only the curated ``repo_ids``.
    Now it keeps ALL repos returned by the listing, so ``fetch_catalog`` can
    synthesise rows for models that aren't in ``CURATED``.  Curated repos
    that fall outside the top-N window still receive individual ``model_info``
    lookups so they never show zero downloads.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return {}
    api = HfApi(token=token) if token else HfApi()
    out: dict[str, dict[str, Any]] = {}
    wanted = set(repo_ids)
    limit = max(len(repo_ids) + 50, dynamic_limit)
    try:
        # ``list_models`` in huggingface_hub 1.x sorts descending by
        # default when ``sort`` is provided, so no ``direction`` kwarg.
        # ``expand`` is required to surface ``last_modified`` on each
        # row without a per-repo ``model_info`` round-trip.
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
            downloads = int(getattr(info, "downloads", 0) or 0)
            lm_raw = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
            lm = ""
            if lm_raw is not None:
                try:
                    lm = lm_raw.isoformat() if hasattr(lm_raw, "isoformat") else str(lm_raw)
                except Exception:  # noqa: BLE001
                    lm = ""
            out[rid] = {"downloads": downloads, "last_modified": lm}
    except Exception as exc:  # noqa: BLE001
        logger.debug("HF list_models failed: %s", exc)

    # Curated rows that fell outside the top-N download window get a
    # cheap per-repo lookup so they don't show 0 downloads forever.
    missing = [rid for rid in wanted if rid not in out]
    if missing:
        for rid in missing:
            try:
                info = api.model_info(
                    rid,
                    expand=["downloads", "lastModified"],
                )
                downloads = int(getattr(info, "downloads", 0) or 0)
                lm_raw = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
                lm = ""
                if lm_raw is not None:
                    try:
                        lm = lm_raw.isoformat() if hasattr(lm_raw, "isoformat") else str(lm_raw)
                    except Exception:  # noqa: BLE001
                        lm = ""
                out[rid] = {"downloads": downloads, "last_modified": lm}
            except Exception as exc:  # noqa: BLE001
                logger.debug("HF model_info(%s) failed: %s", rid, exc)
                out[rid] = {"downloads": 0, "last_modified": ""}
    return out


def _append_dynamic_rows(
    rows: list[CatalogRow],
    live: dict[str, dict[str, Any]],
    curated_ids: set[str],
) -> None:
    """Infer and append rows for discovered models not already in ``rows``."""
    for rid, meta in live.items():
        if rid in curated_ids:
            continue
        inferred = _infer_row_from_hub(
            rid,
            downloads=int(meta.get("downloads", 0) or 0),
            last_modified=str(meta.get("last_modified", "")),
        )
        if inferred is not None:
            rows.append(inferred)


def _is_discovery_complete(cached: dict[str, dict[str, Any]]) -> bool:
    """Return True if the cache contains substantially more than curated rows.

    A cache written before dynamic discovery was implemented (or one where
    the bulk listing failed) only has ``len(CURATED)`` entries.  We treat
    that as incomplete so we try the background enrichment again.
    """
    return len(cached) > len(CURATED) + 5


# ---------------------------------------------------------------------------
# Background enrichment
# ---------------------------------------------------------------------------
# The HF ``list_models`` call with a large limit can take 10–30 s.  We run
# it in a background asyncio task so ``fetch_catalog`` can return the curated
# list immediately and the UI updates automatically once the task finishes.

_enrichment_task: asyncio.Task | None = None  # type: ignore[type-arg]


def is_enriching() -> bool:
    """Return True while the background HF discovery task is in flight."""
    return _enrichment_task is not None and not _enrichment_task.done()


async def _run_enrichment(repo_ids: list[str], token: str | None) -> None:
    global _enrichment_task  # noqa: PLW0603
    try:
        live = await asyncio.to_thread(_enrich_blocking, repo_ids, token)
        if live:
            _save_cache(live)
            logger.info(
                "Background HF enrichment complete: %d repos cached (%d dynamic)",
                len(live),
                len(live) - len(CURATED),
            )
        else:
            logger.debug("Background HF enrichment returned no data")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Background HF enrichment failed: %s", exc)


def _start_enrichment(repo_ids: list[str], token: str | None) -> None:
    """Schedule a background enrichment task unless one is already running."""
    if is_enriching():
        return
    try:
        loop = asyncio.get_running_loop()
        global _enrichment_task  # noqa: PLW0603
        _enrichment_task = loop.create_task(
            _run_enrichment(repo_ids, token),
            name="mlx_catalog_enrichment",
        )
    except RuntimeError:
        pass  # No running loop (e.g. test / CLI context).


async def fetch_catalog(*, token: str | None = None, force: bool = False) -> list[CatalogRow]:
    """Return the curated catalog merged with dynamically discovered models.

    Non-force (normal page load):
      * Returns immediately from the on-disk cache when it exists and is
        both fresh (< 24 h) AND contains more than the curated baseline
        (i.e. dynamic discovery previously succeeded).
      * Otherwise returns the curated list instantly and fires a background
        task to fetch the full mlx-community listing from HF Hub.  The cache
        is written once the background task completes; the next request (or a
        manual Refresh) will pick up the full list.

    Force (user clicked Refresh):
      * Does a blocking HF fetch, writes the cache, and returns the full list.
    """
    curated_ids = {r.repo_id for r in CURATED}
    rows = [CatalogRow(**asdict(r)) for r in CURATED]  # cheap deep copy
    repo_ids = [r.repo_id for r in rows]

    if force:
        # Blocking refresh — user explicitly asked for updated data.
        try:
            live = await asyncio.to_thread(_enrich_blocking, repo_ids, token)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Forced enrichment failed: %s", exc)
            live = {}
        if live:
            _save_cache(live)
        else:
            live = _load_cache() or {}
    else:
        cached = _load_cache()
        if cached is not None and _is_discovery_complete(cached):
            live = cached
        else:
            # Cache is missing or only has curated models — serve curated
            # immediately and kick off background discovery.
            live = cached or {}
            _start_enrichment(repo_ids, token)

    for r in rows:
        meta = live.get(r.repo_id, {})
        r.downloads = int(meta.get("downloads", 0) or 0)
        r.last_modified = str(meta.get("last_modified", ""))

    _append_dynamic_rows(rows, live, curated_ids)
    return rows


# ---------------------------------------------------------------------------
# Live on-demand search (search-as-you-type fallthrough)
# ---------------------------------------------------------------------------
# fetch_catalog() only surfaces the curated list plus the top-N most-downloaded
# mlx-community repos, so niche / brand-new / large models never appear.  When
# the user types a query the UI can't satisfy locally, search_catalog() asks
# the HF Hub directly so any mlx-community repo becomes reachable.


def _search_hub_blocking(
    query: str,
    token: str | None,
    *,
    limit: int = 40,
) -> list[CatalogRow]:
    """Search mlx-community on the HF Hub and return inferred rows.

    Architecture fields default to 0 (so fit scoring is rougher than curated
    rows), but every returned repo is scoreable and loadable.  Repos that look
    like non-LLMs (embedders, ASR, adapters, …) are dropped by
    :func:`_infer_row_from_hub`.
    """
    q = (query or "").strip()
    if not q:
        return []
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []
    api = HfApi(token=token) if token else HfApi()
    rows: list[CatalogRow] = []
    seen: set[str] = set()
    try:
        listing = api.list_models(
            author="mlx-community",
            search=q,
            sort="downloads",
            limit=max(1, min(100, limit)),
            expand=["downloads", "lastModified"],
        )
        for info in listing:
            rid = getattr(info, "id", "") or getattr(info, "modelId", "")
            if not rid or rid in seen:
                continue
            seen.add(rid)
            downloads = int(getattr(info, "downloads", 0) or 0)
            lm_raw = getattr(info, "last_modified", None) or getattr(info, "lastModified", None)
            lm = ""
            if lm_raw is not None:
                try:
                    lm = lm_raw.isoformat() if hasattr(lm_raw, "isoformat") else str(lm_raw)
                except Exception:  # noqa: BLE001
                    lm = ""
            inferred = _infer_row_from_hub(rid, downloads=downloads, last_modified=lm)
            if inferred is not None:
                rows.append(inferred)
    except Exception as exc:  # noqa: BLE001
        logger.debug("HF list_models(search=%r) failed: %s", q, exc)
    return rows


async def search_catalog(
    query: str,
    *,
    token: str | None = None,
    limit: int = 40,
) -> list[CatalogRow]:
    """Live-search the HF Hub for mlx-community models matching ``query``.

    Runs the blocking HF call in a thread so it never blocks the event loop.
    Returns inferred (non-curated) rows; the caller scores them against the
    local hardware via :func:`score_catalog`.
    """
    return await asyncio.to_thread(_search_hub_blocking, query, token, limit=limit)


__all__ = [
    "CatalogRow",
    "CURATED",
    "estimate_footprint_gb",
    "score_row",
    "score_catalog",
    "fetch_catalog",
    "search_catalog",
    "is_enriching",
]


# Suppress unused-import lints when math import becomes optional.
_ = math
