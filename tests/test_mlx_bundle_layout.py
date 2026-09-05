"""Guards for the frozen-app MLX layout assumed by ``scripts/build_backend.py``.

PyInstaller promotes ``libmlx.dylib`` to ``_internal/`` and then
``_relocate_metallib`` copies ``mlx.metallib`` next to it.  If an MLX
wheel moves either file, GPU init in the packaged app fails with
"Failed to load the default metallib".
"""

from __future__ import annotations

import importlib.metadata as metadata
import inspect
import platform
from pathlib import Path

import pytest


def test_huggingface_hub_download_hooks_still_exist():
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import filter_repo_objects

    assert "tqdm_class" in inspect.signature(snapshot_download).parameters
    kept = list(
        filter_repo_objects(
            items=["a.safetensors", "a.bin", "original/x"],
            allow_patterns=["*.safetensors"],
            ignore_patterns=["original/*"],
        )
    )
    assert kept == ["a.safetensors"]
    assert callable(try_to_load_from_cache)
    assert issubclass(HFValidationError, Exception)


@pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine().lower() != "arm64",
    reason="mlx-metal wheel is Apple Silicon only",
)
def test_metallib_sits_beside_libmlx_in_the_wheel():
    dist = metadata.distribution("mlx-metal")
    metallib = Path(dist.locate_file("mlx/lib/mlx.metallib"))
    dylib = Path(dist.locate_file("mlx/lib/libmlx.dylib"))
    assert metallib.is_file(), f"missing {metallib}"
    assert dylib.is_file(), f"missing {dylib}"
    assert metallib.parent == dylib.parent
    assert metallib.name == "mlx.metallib"
    assert dylib.name == "libmlx.dylib"
