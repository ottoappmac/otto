"""Curated Jev-compatible decision models for the Decisions settings tab.

Rows are the MLX builds that ``razorback16/openjev`` serves through
``OPENJEV_BACKEND=mlx``. APUS and the small encoder models are a different
serving stack and are not listed here.

Fit is scored against the machine probe in
:func:`backend.setup_capabilities.probe_capabilities`. The number that
matters is the measured resident set of the sidecar (weights plus the
capped MLX buffer pool), because it has to sit beside the chat model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from backend.setup_capabilities import probe_capabilities


@dataclass(frozen=True)
class DecisionModel:
    repo_id: str
    display_name: str
    quant: str
    weights_gb: float
    # Resident set with OPENJEV_MLX_CACHE_LIMIT_GB=4. The 4-bit figure is
    # measured on an M5 Max; the others add a small working-set margin
    # on top of the Hub weight size.
    resident_gb: float
    license: str
    blurb: str
    featured: bool = False


CURATED: tuple[DecisionModel, ...] = (
    DecisionModel(
        repo_id="mlx-community/diffusiongemma-26B-A4B-it-4bit",
        display_name="DiffusionGemma 26B-A4B (4-bit)",
        quant="4bit",
        weights_gb=16.58,
        resident_gb=18.0,
        license="Apache-2.0",
        blurb="Default. About 0.14 s per decision once warm, about 18 GB resident.",
        featured=True,
    ),
    DecisionModel(
        repo_id="mlx-community/diffusiongemma-26B-A4B-it-8bit",
        display_name="DiffusionGemma 26B-A4B (8-bit)",
        quant="8bit",
        weights_gb=28.0,
        resident_gb=32.0,
        license="Apache-2.0",
        blurb="Closer to the full weights. Needs a machine that can spare about 32 GB.",
    ),
    DecisionModel(
        repo_id="mlx-community/diffusiongemma-26B-A4B-it-bf16",
        display_name="DiffusionGemma 26B-A4B (bf16)",
        quant="bf16",
        weights_gb=51.68,
        resident_gb=56.0,
        license="Apache-2.0",
        blurb="Full precision. Only for machines with a large amount of free memory.",
    ),
)

_BY_ID = {row.repo_id: row for row in CURATED}


def get_model(repo_id: str) -> DecisionModel | None:
    return _BY_ID.get((repo_id or "").strip())


def score_model(
    row: DecisionModel,
    *,
    ram_gb: float,
    wired_limit_gb: float,
    free_disk_gb: float,
) -> dict:
    """Label ``comfortable``, ``tight``, or ``over``.

    Comfortable leaves at least 65% of RAM for the chat model and the OS.
    Tight still leaves a 6 GB floor so a small chat model can load.
    Over cannot sit beside a chat model, which is the 16 GB Mac case for
    the 4-bit build (18 GB resident).
    """
    resident = row.resident_gb
    if ram_gb <= 0:
        fits = "unknown"
    elif resident + 6.0 > ram_gb or (wired_limit_gb > 0 and resident > wired_limit_gb):
        fits = "over"
    elif resident <= ram_gb * 0.35:
        fits = "comfortable"
    else:
        fits = "tight"
    disk_ok = free_disk_gb <= 0 or row.weights_gb + 1.0 <= free_disk_gb
    return {
        "fits": fits,
        "disk_ok": disk_ok,
        "resident_gb": resident,
        "ram_gb": ram_gb,
        "free_disk_gb": free_disk_gb,
    }


def _cache_flags(repo_id: str) -> tuple[bool, bool]:
    """``(already_cached, cache_complete)`` from the local Hub snapshot.

    A directory with no finished snapshot is an interrupted download, the
    same signal the open-weight picker uses for "Continue download".
    """
    from backend.decision_runtime import hf_repo_dir, model_cached

    complete = model_cached(repo_id)
    present = complete or hf_repo_dir(repo_id).is_dir()
    return present, complete


def scored_catalog(hub_cache: str) -> list[dict]:
    """Curated rows plus a fit label and cache state for this machine."""
    caps = probe_capabilities(hub_cache)
    out: list[dict] = []
    for row in CURATED:
        scored = score_model(
            row,
            ram_gb=float(caps.get("ram_gb") or 0),
            wired_limit_gb=float(caps.get("wired_limit_gb") or 0),
            free_disk_gb=float(caps.get("free_disk_gb") or 0),
        )
        already_cached, cache_complete = _cache_flags(row.repo_id)
        out.append({
            **asdict(row),
            **scored,
            "already_cached": already_cached,
            "cache_complete": cache_complete,
        })
    return out
