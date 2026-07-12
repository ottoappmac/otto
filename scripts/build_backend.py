#!/usr/bin/env python3
"""Build the FastAPI backend into a standalone executable using PyInstaller.

Usage:
    python scripts/build_backend.py

We use PyInstaller's ``--onedir`` mode (not ``--onefile``).  ``--onefile``
ships a self-extracting archive that unpacks ~170 MB of dependencies into
a fresh ``/var/folders/.../_MEIxxxx`` temp directory on every launch, which
adds 5-15 seconds of dead time before Python even starts (and triggers a
Gatekeeper re-scan on macOS).  ``--onedir`` keeps the unpacked tree inside
the .app bundle so launch is essentially instant for the bootstrap phase.

The directory is placed at:
  - app/src-tauri/resources/backend/<exe>  — Tauri bundles this whole dir
    into Otto.app/Contents/Resources/backend/ and ``lib.rs`` spawns the exe
    from there at runtime.

Module layout
-------------

The PyInstaller invocation is built up from a series of small ``_*_args``
helpers, each owning a logical group of flags (web framework, LLM
providers, MLX runtime, …).  This keeps the giant flag list reviewable —
adding or removing a dependency only touches the relevant helper, and
each group carries a comment explaining *why* its members are bundled
or excluded.

When adding a new ``--exclude-module``, first ``rg`` the package name
across ``backend/`` and ``src/``.  If anything imports it (directly or
via an ``__init__.py`` chain), excluding it will crash the bundle at
runtime — see the MLX/Playwright/langchain_community history in git
log for examples.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND_ENTRY = ROOT / "backend" / "server.py"
DIST_DIR = ROOT / "dist"
TAURI_BIN_DIR = ROOT / "app" / "src-tauri" / "binaries"
TAURI_RESOURCE_DIR = ROOT / "app" / "src-tauri" / "resources" / "backend"

SEP = ";" if platform.system() == "Windows" else ":"
IS_MACOS_ARM = (
    platform.system() == "Darwin" and platform.machine().lower() == "arm64"
)


def get_target_triple() -> str:
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin":
        arch = "aarch64" if machine == "arm64" else "x86_64"
        return f"{arch}-apple-darwin"
    if system == "Windows":
        arch = "aarch64" if machine == "arm64" else "x86_64"
        return f"{arch}-pc-windows-msvc"
    arch = "x86_64" if machine in ("x86_64", "amd64") else machine
    return f"{arch}-unknown-linux-gnu"


# ── PyInstaller argument groups ──────────────────────────────────────────────


def _web_framework_args() -> list[str]:
    """FastAPI + Uvicorn + ASGI runtime.

    ``uvloop`` is bundled because without it uvicorn falls back to the
    stdlib SelectorEventLoop, which cannot keep up with concurrent LLM
    streams and causes thread-pool deadlocks under load.
    """
    return [
        "--collect-all", "fastapi",
        "--collect-all", "starlette",
        "--collect-all", "uvicorn",
        "--hidden-import", "uvicorn.logging",
        "--hidden-import", "uvicorn.protocols",
        "--hidden-import", "uvicorn.protocols.http",
        "--hidden-import", "uvicorn.protocols.http.auto",
        "--hidden-import", "uvicorn.protocols.websockets",
        "--hidden-import", "uvicorn.protocols.websockets.auto",
        "--hidden-import", "uvicorn.lifespan",
        "--hidden-import", "uvicorn.lifespan.on",
        "--collect-all", "uvloop",
        "--hidden-import", "websockets",
    ]


def _async_http_args() -> list[str]:
    """HTTP/async + SQLite + multipart upload support."""
    return [
        "--collect-all", "httpx",
        "--collect-all", "httpcore",
        "--collect-all", "anyio",
        "--hidden-import", "aiosqlite",
        "--hidden-import", "multipart",
        "--hidden-import", "python_multipart",
    ]


def _pydantic_args() -> list[str]:
    """Pydantic + its native pydantic_core extension."""
    return [
        "--collect-all", "pydantic",
        "--collect-all", "pydantic_core",
    ]


def _llm_provider_args() -> list[str]:
    """LangChain + LangGraph stack used by every chat model.

    ``langchain_community`` is required for the MLX chat-model wrappers
    (``chat_models.mlx.chat_mlx_no_stream.ChatMLXNoStream`` and
    ``chat_models.mlx.command_r.CommandRMLXChat``).  The explicit
    ``--hidden-import`` lines below ensure PyInstaller's submodule
    walker reaches the exact paths those wrappers depend on, even if
    the broader ``--collect-all`` traversal short-circuits.

    ``numpy`` is required because ``langchain_aws/__init__.py``
    eagerly imports ``BedrockEmbeddings`` which transitively imports
    numpy — every Bedrock session would crash without it.
    """
    return [
        "--collect-all", "langchain",
        "--collect-all", "langchain_core",
        "--collect-all", "langchain_anthropic",
        "--collect-all", "langchain_aws",
        "--collect-all", "langchain_openai",
        "--collect-all", "langchain_cohere",
        "--collect-all", "langgraph",
        "--collect-all", "langchain_mcp_adapters",
        "--collect-all", "langchain_community",
        "--hidden-import", "langchain_community.chat_models.mlx",
        "--hidden-import", "langchain_community.llms.mlx_pipeline",
        "--collect-all", "numpy",
    ]


def _mcp_protocol_args() -> list[str]:
    """MCP protocol bindings.

    We deliberately avoid ``--collect-all mcp`` because it scans
    ``mcp.cli`` which has a hard dependency on ``typer``; we ship only
    the runtime-relevant subpackages.
    """
    return [
        "--hidden-import", "mcp",
        "--collect-data", "mcp",
        "--collect-submodules", "mcp.client",
        "--collect-submodules", "mcp.server",
        "--collect-submodules", "mcp.shared",
        "--collect-submodules", "mcp.types",
    ]


def _document_parser_args() -> list[str]:
    """Lightweight document parsers used by ``tools.research.doc_reader``.

    These replace the (much heavier) ``unstructured`` stack — see the
    matching ``--exclude-module unstructured*`` entries below.
    """
    return [
        "--collect-all", "pypdf",
        "--collect-all", "docx",
        "--collect-all", "pptx",
        "--collect-all", "openpyxl",
        "--collect-all", "lxml",
    ]


def _src_package_args() -> list[str]:
    """``src/`` packages the backend imports lazily at runtime.

    The backend's static import graph doesn't reach these modules —
    they're loaded through deferred imports inside route handlers and
    factory functions.  Each needs to be hinted explicitly so the
    PyInstaller bytecode walker picks them up.

    ``--add-data src/`` ships the source tree itself; the frozen
    backend prepends it to ``sys.path`` at startup so these modules
    resolve normally.
    """
    return [
        "--add-data", f"{ROOT / 'src'}{SEP}src",
        "--add-data", f"{ROOT / 'backend' / 'builtin_mcps'}{SEP}backend/builtin_mcps",
        "--hidden-import", "deep_agent",
        "--hidden-import", "deep_agent.model_factory",
        "--hidden-import", "deep_agent.tool_factory",
        "--hidden-import", "tools",
        "--hidden-import", "tools.anthropic",
        "--hidden-import", "tools.anthropic.mcps",
        "--hidden-import", "tools.navigation",
        "--hidden-import", "tools.navigation.web",
        "--hidden-import", "tools.navigation.web.playwright_mcp",
        "--hidden-import", "tools.evaluation",
        "--hidden-import", "tools.evaluation.evaluators",
        "--hidden-import", "tools.evaluation.mcp_server",
        "--hidden-import", "tools.transcripts",
        "--hidden-import", "tools.transcripts.claude_mcp_server",
        "--hidden-import", "tools.transcripts.openclaw_mcp_server",
        "--hidden-import", "tools.transcripts.eval_persistence",
        "--hidden-import", "tools.transcripts.parsers",
        "--hidden-import", "tools.transcripts.parsers.base",
        "--hidden-import", "tools.transcripts.parsers.claude_code",
        "--hidden-import", "tools.transcripts.parsers.cowork",
        "--hidden-import", "tools.transcripts.parsers.openclaw",
        "--hidden-import", "tools.transcripts.parsers._utils",
        "--hidden-import", "utilities",
        "--hidden-import", "utilities.environment",
        "--hidden-import", "utilities.logger",
        "--hidden-import", "agents",
        "--hidden-import", "callbacks",
    ]


def _ancillary_runtime_args() -> list[str]:
    """Miscellaneous runtime helpers.

    SSH (asyncssh) is used by the OpenClaw transcript loader; rank_bm25
    powers research chunk ranking; apscheduler runs background triggers;
    deepagents is the orchestrator core; the rest are small helpers.

    ``youtube_transcript_api`` is imported lazily inside
    ``langchain_community.document_loaders.youtube`` (so PyInstaller's
    static analysis misses it) but is required by the
    ``youtube_transcript`` research tool.
    """
    return [
        "--collect-all", "asyncssh",
        "--collect-all", "rank_bm25",
        "--collect-all", "youtube_transcript_api",
        "--hidden-import", "defusedxml",   # transitive dep of youtube_transcript_api
        "--collect-all", "apscheduler",
        "--collect-all", "deepagents",
        "--hidden-import", "tzlocal",
        "--hidden-import", "jwt",
        "--hidden-import", "certifi",
        "--hidden-import", "dotenv",
    ]


def _mlx_args() -> list[str]:
    """Apple Silicon MLX runtime — bundled for ``provider=mlx`` subagents.

    Adds ~350 MB to the .app bundle.  Skipped on non-arm64-macOS hosts
    where the MLX wheels aren't installable (CI Linux builds, Intel
    Mac, Windows).

    ``mlx`` ships a Metal shader library (``mlx.metallib``) that needs
    a post-build relocation — see :func:`_relocate_metallib`.  Both
    ``mlx_lm`` and ``mlx_vlm`` use dynamic dispatch over per-architecture
    model modules, so we bundle them whole rather than enumerating
    hidden-imports per model family.

    Transitive hard deps that MUST be bundled alongside MLX:

    - ``transformers`` — mlx_lm's ``tokenizer_utils.py`` does an eager
      ``from transformers import AutoTokenizer, PreTrainedTokenizerFast``
      at module load.
    - ``tokenizers`` / ``safetensors`` / ``huggingface_hub`` —
      transformers' hard runtime deps (Rust-backed; native binaries).
    - ``sentencepiece`` / ``google.protobuf`` — mlx_lm hard deps used
      by various model architectures' tokenizers.

    mlx_vlm additionally requires ``cv2`` (opencv-python), ``datasets``,
    and ``miniaudio`` for its full feature surface — those are NOT
    bundled here because they add another ~300 MB and the only call
    site (``chat_models.mlx.chat_vlm.MLXVLChatModel``) imports them
    lazily inside methods.  If a user actually runs the MLX VLM path,
    they will see a clear ``ModuleNotFoundError`` at first inference
    pointing at the missing dep, rather than paying the bundle cost
    up-front for everyone.

    ``mlx_whisper`` powers on-device speech-to-text
    (``backend.voice.stt`` → system-audio + mic transcription).  Its
    ``transcribe`` entrypoint eagerly imports ``timing.py`` (``numba``
    + ``scipy.signal``) and ``tokenizer.py`` (``tiktoken``) at module
    load, so all of those must be bundled too — otherwise
    ``import mlx_whisper`` raises ``ImportError`` inside the frozen
    backend and every transcription fails with a misleading
    "mlx-whisper is not installed" message.  ``numba``/``llvmlite``/
    ``scipy`` are therefore NOT in ``_exclude_args`` — keep those two
    lists in sync.  ``torch`` is intentionally left out: it appears in
    mlx_whisper's ``Requires-Dist`` but is only touched by
    ``torch_whisper.py`` (OpenAI-checkpoint conversion), which the
    ``transcribe`` path never imports.
    """
    if not IS_MACOS_ARM:
        return []
    return [
        "--collect-all", "mlx",
        "--collect-all", "mlx_lm",
        "--collect-all", "mlx_vlm",
        "--collect-all", "mlx_whisper",
        "--collect-all", "transformers",
        "--collect-all", "tokenizers",
        "--collect-all", "safetensors",
        "--collect-all", "huggingface_hub",
        "--collect-all", "sentencepiece",
        "--collect-all", "google.protobuf",
        "--collect-all", "numba",
        "--collect-all", "llvmlite",
        "--collect-all", "scipy",
        "--collect-all", "tiktoken",
        "--collect-submodules", "tiktoken_ext",
    ]


def _embedding_args() -> list[str]:
    """sqlite-vec extension + sentence-transformers bundle args.

    sqlite-vec ships a native C extension (``_sqlite_vec.so``); ``--collect-all``
    ensures the .so lands in the bundle.  sentence-transformers is cross-platform
    so no platform guard is needed.
    """
    return [
        "--collect-all", "sqlite_vec",
        "--collect-all", "sentence_transformers",
        "--hidden-import", "tools.research.semantic_search",
        "--hidden-import", "backend.embedding_index",
        "--hidden-import", "backend.routes.embeddings",
    ]


def _evaluation_args() -> list[str]:
    """DeepEval end-of-run evaluation stack (``tools.evaluation`` + ``eval_runner``).

    DeepEval ships **on-disk data files** — metric prompt templates and
    benchmark shot/CoT prompts — that its metric classes read at runtime
    (e.g. ``deepeval/templates/metrics/templates.json``).  PyInstaller's
    static walk picks up the ``.py`` modules but not this package data, so
    without ``--collect-all`` the first metric evaluation in the packaged
    app crashes with ``FileNotFoundError: … /deepeval/templates/…``.

    ``--collect-all`` (rather than a bare ``--collect-data``) is used
    because the metric classes are imported lazily inside
    ``tools.evaluation.evaluators`` (``from deepeval.metrics import …``
    deferred into factory functions), so the submodule graph must be
    collected explicitly too — the static bytecode walker never reaches
    those deferred import sites.
    """
    return [
        "--collect-all", "deepeval",
    ]


def _exclude_args() -> list[str]:
    """Packages PyInstaller would otherwise drag in via its static walk.

    Every name here was confirmed (via grep over ``backend/`` and
    ``src/``) to have zero non-test import sites.  They live in the
    venv only because optional langchain extras list them as deps.
    Each exclusion saves tens of MB and shaves startup time.

    DO NOT add a name here without first confirming nothing imports it.
    An incorrect exclusion crashes the bundle at runtime — see the MLX,
    Playwright, and langchain_community regressions for prior art.
    """
    return [
        # Playwright — the Python backend never imports the
        # ``playwright`` package directly.  Browser automation goes
        # through the external Node MCP service (``@playwright/mcp``)
        # on port 8931, which the backend talks to over HTTP.
        #
        # NOTE: ``tools.navigation.{__init__,web/__init__}.py`` lazy-
        # load the Playwright-using submodules so a bare
        # ``from tools.navigation.computer import …`` does NOT trigger
        # an import of the missing ``playwright`` package.
        # ``agents/web_voyager.py`` also lazy-imports
        # ``PlaywrightComputerUseNavigator`` inside ``__init__`` so
        # ``from agents import WebVoyagerGraph`` is safe at module load.
        # If you change any of these, re-verify this exclusion.
        "--exclude-module", "playwright",
        # Document-parsing stack we replaced with lighter alternatives
        # (see ``_document_parser_args``).
        "--exclude-module", "unstructured",
        "--exclude-module", "unstructured_client",
        "--exclude-module", "langchain_unstructured",
        "--exclude-module", "unstructured_inference",
        "--exclude-module", "magic",
        # Heavy ML training/inference frameworks pulled in transitively
        # by various langchain extras — nothing in the backend uses
        # them.  Excluding saves >1 GB total.
        #
        # NOTE: ``transformers`` is intentionally NOT excluded — it is
        # a hard runtime dep of ``mlx_lm`` (see ``_mlx_args``).  Adding
        # it here will break MLX subagents at first inference.
        "--exclude-module", "torch",
        "--exclude-module", "torchvision",
        "--exclude-module", "torchaudio",
        "--exclude-module", "accelerate",
        "--exclude-module", "timm",
        "--exclude-module", "onnxruntime",
        "--exclude-module", "onnx",
        "--exclude-module", "tensorflow",
        "--exclude-module", "jax",
        # NLP / scientific stack — transitive only.
        #
        # NOTE: ``numba``, ``llvmlite``, and ``scipy`` are intentionally
        # NOT excluded — ``mlx_whisper.transcribe`` imports them eagerly
        # (via ``timing.py``) and they are bundled in ``_mlx_args``.
        # Excluding them here would win over ``--collect-all`` and break
        # speech-to-text in the frozen backend.
        "--exclude-module", "spacy",
        "--exclude-module", "thinc",
        "--exclude-module", "sympy",
        "--exclude-module", "matplotlib",
        "--exclude-module", "pandas",
        "--exclude-module", "cv2",
        "--exclude-module", "sklearn",
        "--exclude-module", "scikit_learn",
        "--exclude-module", "networkx",
        # Vector stores reachable through ``langchain_community`` but
        # never actually instantiated by the backend.  Excluding them
        # is what lets us bundle the rest of ``langchain_community``
        # without a ~200 MB Chroma/Arrow blob.
        "--exclude-module", "chromadb",
        "--exclude-module", "chromadb_rust_bindings",
        "--exclude-module", "langchain_chroma",
        "--exclude-module", "pyarrow",
        # Misc transitive bloat we don't use.
        #
        # NOTE: ``google.protobuf`` is intentionally NOT excluded —
        # ``transformers`` and ``mlx_lm`` use it for tokenizer model
        # serialisation.  See ``_mlx_args`` for the matching include.
        "--exclude-module", "kubernetes",
        "--exclude-module", "grpc",
        "--exclude-module", "datasets",
        "--exclude-module", "pypandoc",
        "--exclude-module", "fontTools",
        "--exclude-module", "debugpy",
    ]


# ── Build orchestration ──────────────────────────────────────────────────────


def _build_pyinstaller_command(build_name: str) -> list[str]:
    """Compose the full PyInstaller invocation as an ordered argv list."""
    return [
        sys.executable, "-m", "PyInstaller",
        "--onedir",
        "--noconfirm",
        "--clean",
        "--name", build_name,
        "--distpath", str(DIST_DIR),
        "--specpath", str(ROOT / "build"),
        *_web_framework_args(),
        *_async_http_args(),
        *_pydantic_args(),
        *_llm_provider_args(),
        *_mcp_protocol_args(),
        *_document_parser_args(),
        *_src_package_args(),
        *_ancillary_runtime_args(),
        *_mlx_args(),
        *_embedding_args(),
        *_evaluation_args(),
        *_exclude_args(),
        str(BACKEND_ENTRY),
    ]


def _relocate_metallib(dist_tree: Path) -> None:
    """Place ``mlx.metallib`` next to the runtime ``libmlx.dylib``.

    PyInstaller promotes ``libmlx.dylib`` to ``_internal/`` so the
    ``@rpath/libmlx.dylib`` baked into ``mlx/core.*.so`` resolves, but
    it leaves the Metal shader library (``mlx.metallib``) at
    ``_internal/mlx/lib/``.  MLX's C++ runtime locates the metallib by
    calling ``dladdr()`` on a symbol inside ``libmlx.dylib`` and
    searching the *same directory*, so the file needs to sit at
    ``_internal/mlx.metallib`` for GPU init to succeed.  Without this
    step, the first ``mx.compile`` call raises "Failed to load the
    default metallib".
    """
    src = dist_tree / "_internal" / "mlx" / "lib" / "mlx.metallib"
    dst = dist_tree / "_internal" / "mlx.metallib"
    if not src.exists():
        return
    if dst.exists():
        print(f"mlx.metallib already in place at {dst}")
        return
    shutil.copy2(src, dst)
    print(f"Relocated mlx.metallib -> {dst}")


def _publish_to_tauri(dist_tree: Path, build_name: str, ext: str) -> Path:
    """Copy the onedir tree into Tauri's resource folder.

    The copy normalises the executable name from
    ``otto-backend-<triple>`` to ``otto-backend`` so ``lib.rs`` can
    spawn a single stable filename regardless of build host.  Returns
    the resolved exe path inside the resource folder.
    """
    if TAURI_RESOURCE_DIR.exists():
        shutil.rmtree(TAURI_RESOURCE_DIR)
    TAURI_RESOURCE_DIR.mkdir(parents=True, exist_ok=True)

    for entry in dist_tree.iterdir():
        dest = TAURI_RESOURCE_DIR / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest)
        else:
            shutil.copy2(entry, dest)

    resource_exe = TAURI_RESOURCE_DIR / ("otto-backend" + ext)
    bundled_exe = TAURI_RESOURCE_DIR / (build_name + ext)
    if bundled_exe.exists() and bundled_exe != resource_exe:
        if resource_exe.exists():
            resource_exe.unlink()
        bundled_exe.rename(resource_exe)

    # The legacy sidecar/binaries dir is unused now that we ship a
    # full ``--onedir`` tree; clean it up to keep the .app tidy.
    if TAURI_BIN_DIR.exists():
        for stale in TAURI_BIN_DIR.iterdir():
            if stale.is_file():
                stale.unlink()

    return resource_exe


def build() -> None:
    host_triple = get_target_triple()
    target_triple = os.environ.get("BACKEND_TARGET_TRIPLE") or host_triple
    ext = ".exe" if platform.system() == "Windows" else ""
    build_name = f"otto-backend-{host_triple}"

    print(f"Building backend for {target_triple} (host: {host_triple})")
    if IS_MACOS_ARM:
        print("Including MLX bundle (~200 MB) — Apple Silicon target")

    cmd = _build_pyinstaller_command(build_name)
    subprocess.run(cmd, check=True, cwd=ROOT)

    dist_tree = DIST_DIR / build_name
    if not dist_tree.is_dir():
        raise RuntimeError(f"Expected --onedir output at {dist_tree}, not found")

    src_exe = dist_tree / (build_name + ext)
    if not src_exe.exists():
        raise RuntimeError(f"Backend executable not found at {src_exe}")

    _relocate_metallib(dist_tree)
    resource_exe = _publish_to_tauri(dist_tree, build_name, ext)

    print(f"Backend tree placed at: {TAURI_RESOURCE_DIR}")
    print(f"Backend executable:     {resource_exe}")


if __name__ == "__main__":
    build()
