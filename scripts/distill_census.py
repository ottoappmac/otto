#!/usr/bin/env python3
"""Count usable Path A trajectories from this machine's session history.

    python scripts/distill_census.py
    python scripts/distill_census.py --persist
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backend.distillation.census import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
