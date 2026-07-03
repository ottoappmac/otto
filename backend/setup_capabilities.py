"""Probe the local machine for the Setup Hub and MLX model-fit scoring.

All probing is read-only and stays on-device.  Output is a single
serialisable dict consumed by:

* ``GET /api/setup/capabilities`` — UI hardware summary on the
  On-Device tab and (later) the first-run overlay.
* ``GET /api/mlx/catalog`` — joins each curated row's footprint
  estimate with these numbers to label models comfortable / tight /
  over the line.

Everything here is stdlib + a single optional ``huggingface_hub``
import for the cache scan.  We avoid ``psutil`` to keep the install
surface small (the codebase already has enough native deps to
worry about).
"""

from __future__ import annotations

import logging
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SetupCapabilities:
    """Snapshot of the local machine, normalised to GB.

    ``wired_limit_gb`` is the rough ceiling for an MLX model on Apple
    Silicon (the GPU wired-memory cap).  On non-Apple hosts we report
    total RAM since there's no equivalent partition.  ``ram_gb`` is the
    physical total; the practical "comfortable" budget is computed in
    :mod:`backend.mlx_catalog` against a configurable fraction.
    """

    platform: str
    arch: str
    apple_silicon: bool
    chip: str
    cpu_brand: str
    ram_gb: float
    free_disk_gb: float
    wired_limit_gb: float
    hf_token_set: bool
    hub_cache_dir: str
    models_cached: int
    models_cached_size_gb: float


# ---------------------------------------------------------------------------
# Low-level probes
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str:
    """Return ``sysctl -n <key>`` stripped, or empty string on any error."""
    if not shutil.which("sysctl"):
        return ""
    try:
        proc = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return (proc.stdout or "").strip()


def _ram_bytes_macos() -> int:
    raw = _sysctl("hw.memsize")
    try:
        return int(raw)
    except ValueError:
        return 0


def _ram_bytes_linux() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                return kb * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _ram_bytes_windows() -> int:
    try:
        import ctypes  # local — Windows-only path

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        class _MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        return int(stat.ullTotalPhys)
    except Exception:  # noqa: BLE001
        return 0


def _ram_bytes() -> int:
    sysname = platform.system()
    if sysname == "Darwin":
        return _ram_bytes_macos()
    if sysname == "Linux":
        return _ram_bytes_linux()
    if sysname == "Windows":
        return _ram_bytes_windows()
    return 0


def _detect_chip() -> tuple[str, str]:
    """Returns ``(friendly_chip_label, raw_cpu_brand)``."""
    sysname = platform.system()
    if sysname == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        m = re.search(r"Apple\s+(M\d+(?:\s+\w+)?)", brand or "")
        if m:
            return (f"Apple {m.group(1).strip()}", brand)
        if platform.machine() == "arm64":
            return ("Apple Silicon", brand or "")
        return ("Intel Mac", brand or "")
    if sysname == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    brand = line.split(":", 1)[1].strip()
                    return (brand, brand)
        except OSError:
            pass
    proc = platform.processor() or platform.machine()
    return (proc, proc)


def _wired_limit_gb_macos(total_ram_gb: float) -> float:
    """Best-effort wired-memory ceiling for the GPU on Apple Silicon.

    macOS's default wired-memory limit is roughly 67% on 8 GB Macs,
    ~75% on 16-32 GB, and ~80% on 64+ GB.  Users can raise it via
    ``sudo sysctl iogpu.wired_limit_mb=...``; we honour the override
    when present.
    """
    raw = _sysctl("iogpu.wired_limit_mb")
    try:
        mb = int(raw)
    except ValueError:
        mb = 0
    if mb > 0:
        return round(mb / 1024.0, 2)
    if total_ram_gb >= 64:
        return round(0.80 * total_ram_gb, 2)
    if total_ram_gb >= 16:
        return round(0.75 * total_ram_gb, 2)
    return round(0.67 * total_ram_gb, 2)


def _free_disk_bytes(target: Path) -> int:
    """``shutil.disk_usage`` walking up to the first existing parent."""
    candidates = [target, target.parent, Path.home(), Path("/")]
    for p in candidates:
        try:
            if p.exists():
                return int(shutil.disk_usage(p).free)
        except OSError:
            continue
    return 0


def _models_cached_summary(hub_cache: Path) -> tuple[int, float]:
    """Number + total size (GB) of repos already in the Hub cache."""
    if not hub_cache.is_dir():
        return (0, 0.0)
    try:
        from huggingface_hub import scan_cache_dir

        info = scan_cache_dir(str(hub_cache))
        n = sum(1 for _ in info.repos)
        sz_gb = sum(repo.size_on_disk for repo in info.repos) / (1024 ** 3)
        return (n, round(sz_gb, 2))
    except Exception as exc:  # noqa: BLE001
        logger.debug("scan_cache_dir failed for %s: %s", hub_cache, exc)
        try:
            return (sum(1 for _ in hub_cache.glob("models--*")), 0.0)
        except OSError:
            return (0, 0.0)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def probe_capabilities(hub_cache: str | Path) -> dict[str, Any]:
    """Return a fresh capabilities snapshot.

    Cheap (a couple of sysctl calls + one dir walk via huggingface_hub).
    Safe to call on every page load.  ``hub_cache`` should be the
    resolved absolute path produced by ``resolve_hf_hub_cache_dir``.
    """
    sysname = platform.system()
    arch = platform.machine()
    apple = sysname == "Darwin" and arch == "arm64"

    ram = _ram_bytes()
    ram_gb = round(ram / (1024 ** 3), 2) if ram else 0.0
    chip, brand = _detect_chip()

    wired_gb = _wired_limit_gb_macos(ram_gb) if apple else ram_gb

    hub = Path(str(hub_cache))
    free_gb = round(_free_disk_bytes(hub) / (1024 ** 3), 2)
    cached_n, cached_gb = _models_cached_summary(hub)

    hf_token_set = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    return asdict(
        SetupCapabilities(
            platform=(sysname or "unknown").lower(),
            arch=arch or "",
            apple_silicon=apple,
            chip=chip,
            cpu_brand=brand,
            ram_gb=ram_gb,
            free_disk_gb=free_gb,
            wired_limit_gb=wired_gb,
            hf_token_set=hf_token_set,
            hub_cache_dir=str(hub),
            models_cached=cached_n,
            models_cached_size_gb=cached_gb,
        )
    )


__all__ = ["SetupCapabilities", "probe_capabilities"]
