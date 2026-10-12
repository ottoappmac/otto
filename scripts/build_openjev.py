#!/usr/bin/env python3
"""Build the OpenJev venv that the GitHub macOS app build embeds.

The packaged app is signed and notarised, so it cannot pip-install this
runtime on first launch: those binaries would be unsigned. This script
installs the pinned ``openjev[mlx]`` into its own relocatable venv at
``app/src-tauri/resources/openjev/`` before ``tauri build``. The sign step
in ``.github/workflows/build-app.yml`` then signs the Mach-O files inside
the finished ``.app``.

The venv is not Otto's. Model weights are not downloaded.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "app" / "src-tauri" / "resources" / "openjev"
sys.path.insert(0, str(ROOT))

from backend.config import DecisionConfig  # noqa: E402
from backend.decision_runtime import _GIT_URL  # noqa: E402


def _build_exo():
    spec = importlib.util.spec_from_file_location(
        "build_exo", ROOT / "scripts" / "build_exo.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("scripts/build_exo.py could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(args: list[str], env: dict[str, str]) -> None:
    print(f"[build_openjev] $ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True, env=env)


def main() -> None:
    machine = platform.machine().lower()
    if platform.system() != "Darwin" or machine not in {"arm64", "aarch64"}:
        raise SystemExit(
            "The OpenJev MLX bundle is built on Apple Silicon macOS only "
            f"(this host is {platform.system()} {platform.machine()})."
        )
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is not on PATH")

    ref = (DecisionConfig().runtime_ref or "").strip()
    if not ref:
        raise SystemExit("DecisionConfig.runtime_ref is empty")

    DEST.mkdir(parents=True, exist_ok=True)
    hidden = DEST / ".venv"
    public = DEST / "venv"
    if hidden.exists():
        shutil.rmtree(hidden)
    if public.exists():
        shutil.rmtree(public)

    env = os.environ.copy()
    # A Homebrew Python hard-codes an absolute Cellar path. uv's own
    # interpreter is the one scripts/build_exo.py can relocate.
    env["UV_PYTHON_PREFERENCE"] = "only-managed"
    env.setdefault("UV_NATIVE_TLS", "1")

    _run([uv, "venv", "--python", "3.11", str(hidden)], env)
    spec = f"openjev[mlx] @ git+{_GIT_URL}@{ref}"
    _run([uv, "pip", "install", "--python", str(hidden / "bin" / "python"), spec], env)

    exo = _build_exo()
    exo.make_relocatable(DEST)
    exo.verify_relocatable(DEST)
    hidden.rename(public)
    (DEST / "INSTALLED_REF").write_text(ref + "\n", encoding="utf-8")
    print(f"[build_openjev] ready at {public} ({ref[:12]})")


if __name__ == "__main__":
    main()
