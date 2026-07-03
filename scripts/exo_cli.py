#!/usr/bin/env python3
"""Thin CLI shim for the exo cluster lifecycle helper.

The real implementation lives in :mod:`backend.exo_cli` so the same code
powers the standalone CLI, the FastAPI routes (``backend/routes/exo.py``),
and the LangChain tools (``backend/exo_tools.py``) without duplication.

Running ``python3 scripts/exo_cli.py up`` from the repo root works
without installing the wheel (the project root is added to ``sys.path``
below). When this script needs to ``scp`` itself to a remote node it is
the implementation file in ``backend/exo_cli.py`` that travels — that
file is fully self-contained (stdlib-only) so it runs on a vanilla
Python 3 install with no Otto codebase present.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.exo_cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
