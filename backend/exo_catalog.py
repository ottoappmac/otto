"""EXO catalog loader + cluster-aware fit scoring.

The cluster's ``GET /v1/models`` is just a flat list of ids — useful to
know *what's available* but not whether anything will actually run on
the user's specific cluster (1× M2 Pro? 4× M3 Ultras? wildly different
budgets).  exo ships richer metadata in
``<exo_repo>/resources/inference_model_cards/*.toml``: every card has
``storage_size.in_bytes``, ``n_layers``, ``hidden_size``,
``num_key_value_heads``, ``family``, ``quantization``, and
``capabilities``.

This module:

* Reads those TOML cards into a list of :class:`ExoCatalogRow`.
* Estimates a runtime footprint per model (weights + KV cache +
  framework overhead).
* Scores each row against the live cluster topology
  (:class:`backend.exo_cli.NodeInfo` from the cluster's ``/state``
  endpoint), returning ``comfortable`` / ``tight`` / ``over`` plus a
  per-node bottleneck so the UI can label rows accurately for any
  ``min_nodes`` setting.

The scoring is intentionally bucketed.  We're not trying to predict
exact memory usage — we're answering "could this load on this
cluster?" with green / amber / red tiers.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Python 3.11+ ships ``tomllib`` in the stdlib; older environments
# fall back to the ``tomli`` shim.
try:
    import tomllib  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Catalog row shape
# ---------------------------------------------------------------------------


@dataclass
class ExoCatalogRow:
    """One model card surfaced from exo's ``inference_model_cards``."""

    model_id: str
    family: str
    base_model: str
    quant: str                 # "4bit" / "8bit" / "bf16" / etc
    weights_gb: float
    params_b: float            # estimated from base_model name when possible
    n_layers: int
    hidden_size: int
    num_kv_heads: int
    context_length: int
    capabilities: list[str] = field(default_factory=list)
    tier: str = "balanced"     # "starter" | "balanced" | "power" | "frontier"
    # Live cluster overlay — populated by :func:`fetch_catalog`.
    downloaded: bool = False
    loaded: bool = False
    featured: bool = False


# ---------------------------------------------------------------------------
# Card loading
# ---------------------------------------------------------------------------


_CARD_GLOB = "*.toml"
_FEATURED: set[str] = {
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Llama-3.3-70B-Instruct-4bit",
    "mlx-community/DeepSeek-V3.2-4bit",
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
}


def _params_b_from_name(base_model: str, weights_gb: float, quant: str) -> float:
    """Best-effort parameter count from the model name.

    Falls back to a back-of-the-envelope estimate from weight size +
    quantisation bits when nothing parses cleanly.  Off by ~10–30% on
    MoE models because we don't have the active-vs-total params split,
    but good enough to display "≈ N B params" in the UI.
    """
    import re

    m = re.search(r"(\d+(?:\.\d+)?)\s*B", base_model)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Back-of-envelope: bytes_per_param ≈ {4bit: 0.5, 8bit: 1.0, bf16: 2.0}.
    bpp = {"4bit": 0.5, "6bit": 0.75, "8bit": 1.0, "bf16": 2.0, "fp16": 2.0}.get(
        quant.lower(), 0.5,
    )
    if bpp <= 0:
        return 0.0
    return round((weights_gb * (1024 ** 3)) / (bpp * 1e9), 1)


def _tier_for(weights_gb: float) -> str:
    if weights_gb < 4:
        return "starter"
    if weights_gb < 20:
        return "balanced"
    if weights_gb < 80:
        return "power"
    return "frontier"


def _parse_card(path: Path) -> ExoCatalogRow | None:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.debug("could not read exo model card %s: %s", path, exc)
        return None

    model_id = str(data.get("model_id") or "").strip()
    if not model_id:
        return None

    storage = data.get("storage_size") or {}
    in_bytes = storage.get("in_bytes")
    weights_gb = round(float(in_bytes) / (1024 ** 3), 2) if in_bytes else 0.0

    quant = str(data.get("quantization") or "").strip() or "unknown"
    base_model = str(data.get("base_model") or model_id).strip()
    family = str(data.get("family") or "").strip() or "other"

    n_layers = int(data.get("n_layers") or 0)
    hidden = int(data.get("hidden_size") or 0)
    n_kv = int(data.get("num_key_value_heads") or 0)
    ctx = int(data.get("context_length") or 32768)
    caps_raw = data.get("capabilities")
    capabilities = [str(x) for x in caps_raw] if isinstance(caps_raw, list) else []

    return ExoCatalogRow(
        model_id=model_id,
        family=family,
        base_model=base_model,
        quant=quant,
        weights_gb=weights_gb,
        params_b=_params_b_from_name(base_model, weights_gb, quant),
        n_layers=n_layers,
        hidden_size=hidden,
        num_kv_heads=n_kv,
        context_length=ctx,
        capabilities=capabilities,
        tier=_tier_for(weights_gb),
        featured=model_id in _FEATURED,
    )


def load_cards(cards_dir: str | Path) -> list[ExoCatalogRow]:
    """Read every ``*.toml`` in ``cards_dir`` into :class:`ExoCatalogRow`.

    Empty list when the directory doesn't exist (e.g. cluster not yet
    provisioned).  Skips cards that fail to parse.
    """
    p = Path(cards_dir)
    if not p.is_dir():
        return []
    rows: list[ExoCatalogRow] = []
    for f in sorted(p.glob(_CARD_GLOB)):
        row = _parse_card(f)
        if row is not None:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Footprint + cluster-aware scoring
# ---------------------------------------------------------------------------


_FRAMEWORK_OVERHEAD_GB = 0.5
_ACTIVATION_OVERHEAD_GB = 1.0


def _kv_dtype_bytes(kv_bits: int | None) -> float:
    """Bytes per KV value at runtime."""
    if kv_bits == 4:
        return 0.5
    if kv_bits == 8:
        return 1.0
    return 2.0  # fp16 / bf16 default


def estimate_footprint_gb(
    row: ExoCatalogRow, *, ctx_len: int, kv_bits: int | None,
) -> dict[str, float]:
    """Rough cluster-wide runtime footprint.

    For dense models the standard formula is::

        kv_bytes = 2 · n_layers · n_kv_heads · head_dim · ctx · kv_bpv

    We use ``head_dim ≈ hidden_size / max(num_kv_heads, 1)`` when
    ``num_key_value_heads`` is present (it is on every card we ship);
    fall back to a flat constant when it's missing so the row still
    scores instead of dropping out of the catalog.
    """
    layers = max(1, row.n_layers or 32)
    kv_heads = max(1, row.num_kv_heads or 8)
    head_dim = max(1, (row.hidden_size or 4096) // kv_heads)
    ctx = max(512, int(ctx_len))
    kv_bpv = _kv_dtype_bytes(kv_bits)

    kv_bytes = 2.0 * layers * kv_heads * head_dim * ctx * kv_bpv
    kv_gb = kv_bytes / (1024 ** 3)

    overhead_gb = _FRAMEWORK_OVERHEAD_GB + _ACTIVATION_OVERHEAD_GB
    if "vision" in row.capabilities:
        overhead_gb += 1.5

    total = row.weights_gb + kv_gb + overhead_gb

    return {
        "weights_gb": round(row.weights_gb, 2),
        "kv_gb": round(kv_gb, 2),
        "overhead_gb": round(overhead_gb, 2),
        "total_gb": round(total, 2),
    }


def _node_budgets(nodes: list[dict[str, Any]]) -> list[float]:
    """Per-node total RAM (GB), descending.

    ``NodeInfo`` is exposed via ``ExoStatus.nodes`` as a list of dicts
    in the API response; we accept both that shape and live
    :class:`backend.exo_cli.NodeInfo` instances.
    """
    out: list[float] = []
    for n in nodes or []:
        if hasattr(n, "memory_total_gb"):
            mem = getattr(n, "memory_total_gb", None)
        else:
            mem = (n or {}).get("memory_total_gb")
        try:
            v = float(mem) if mem is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        if v > 0:
            out.append(v)
    out.sort(reverse=True)
    return out


def score_row(
    row: ExoCatalogRow,
    *,
    nodes: list[dict[str, Any]],
    min_nodes: int = 1,
    ctx_len: int = 8192,
    kv_bits: int | None = None,
    comfortable_fraction: float = 0.5,
    ceiling_fraction: float = 0.85,
) -> dict[str, Any]:
    """Score ``row`` against a cluster topology.

    Sharding is approximated as **even layer split** across ``min_nodes``
    nodes; the bottleneck is therefore the smallest node we'd actually
    use.  This matches what exo's placement does for plain
    pipeline-parallel placements; tensor-parallel within a node is left
    out of the estimate (it only helps for memory and throughput, not
    for "would this load at all").

    Returns the breakdown plus:

    * ``fits``: ``"comfortable"`` / ``"tight"`` / ``"over"`` /
      ``"unknown"`` (when the cluster is offline / has no node info).
    * ``per_node_gb``: bytes each used node would hold.
    * ``bottleneck_gb``: smallest used node's total RAM.
    * ``min_nodes_required``: smallest ``min_nodes`` value that would
      flip this row into ``"comfortable"``, or ``None`` when even
      maxing out the cluster wouldn't help.
    """
    breakdown = estimate_footprint_gb(row, ctx_len=ctx_len, kv_bits=kv_bits)
    total = breakdown["total_gb"]
    budgets = _node_budgets(nodes)

    if not budgets:
        return {
            **breakdown,
            "fits": "unknown",
            "per_node_gb": round(total, 2),
            "bottleneck_gb": 0.0,
            "ceiling_gb": 0.0,
            "min_nodes_required": None,
        }

    n_eff = max(1, min(int(min_nodes), len(budgets)))
    used = budgets[:n_eff]
    bottleneck = used[-1]
    per_node = total / float(n_eff)
    comfortable_budget = comfortable_fraction * bottleneck
    ceiling = ceiling_fraction * bottleneck

    if per_node <= comfortable_budget:
        fits = "comfortable"
    elif per_node <= ceiling:
        fits = "tight"
    else:
        fits = "over"

    # How many nodes (within current cluster size) would we need to
    # fit comfortably?  None means "even using the whole cluster the
    # smallest-node bottleneck still kills it".
    min_required: int | None = None
    for k in range(1, len(budgets) + 1):
        bn = budgets[k - 1]
        if total / float(k) <= comfortable_fraction * bn:
            min_required = k
            break

    return {
        **breakdown,
        "fits": fits,
        "per_node_gb": round(per_node, 2),
        "bottleneck_gb": round(bottleneck, 2),
        "ceiling_gb": round(ceiling, 2),
        "min_nodes_required": min_required,
    }


def score_catalog(
    rows: Iterable[ExoCatalogRow],
    *,
    nodes: list[dict[str, Any]],
    min_nodes: int = 1,
    ctx_len: int = 8192,
    kv_bits: int | None = None,
    comfortable_fraction: float = 0.5,
) -> list[dict[str, Any]]:
    """Score and sort the catalog.

    Order is loaded → comfortable → tight → over, then featured first,
    then by params descending so the user sees the most capable model
    that comfortably fits at the top of each tier.
    """
    out: list[dict[str, Any]] = []
    for r in rows:
        score = score_row(
            r,
            nodes=nodes,
            min_nodes=min_nodes,
            ctx_len=ctx_len,
            kv_bits=kv_bits,
            comfortable_fraction=comfortable_fraction,
        )
        out.append({**asdict(r), **score})

    fit_rank = {"comfortable": 1, "tight": 2, "unknown": 3, "over": 4}
    out.sort(
        key=lambda x: (
            0 if x.get("loaded") else 1,
            0 if x.get("downloaded") else 1,
            fit_rank.get(str(x.get("fits", "")), 9),
            0 if x.get("featured") else 1,
            -float(x.get("params_b", 0) or 0),
            x.get("base_model", ""),
        )
    )
    return out


# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------
#
# The cards directory rarely changes (only on cluster upgrade) so we
# memoise the parse for 5 minutes.  ``invalidate_cache()`` is exposed
# for tests and can be called by the provision job after pulling new
# exo sources.

_cache: dict[str, tuple[float, list[ExoCatalogRow]]] = {}
_CACHE_TTL_S = 300.0


def get_catalog(cards_dir: str | Path, *, force: bool = False) -> list[ExoCatalogRow]:
    key = str(cards_dir)
    now = time.monotonic()
    if not force:
        hit = _cache.get(key)
        if hit and now - hit[0] < _CACHE_TTL_S:
            return [ExoCatalogRow(**asdict(r)) for r in hit[1]]
    rows = load_cards(cards_dir)
    _cache[key] = (now, rows)
    return [ExoCatalogRow(**asdict(r)) for r in rows]


def invalidate_cache() -> None:
    _cache.clear()


__all__ = [
    "ExoCatalogRow",
    "load_cards",
    "estimate_footprint_gb",
    "score_row",
    "score_catalog",
    "get_catalog",
    "invalidate_cache",
]
