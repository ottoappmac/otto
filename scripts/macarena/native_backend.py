#!/usr/bin/env python
"""Start Otto's backend for the MacArena Otto-native track.

Identical to ``python -m backend`` except that, before anything loads, Otto's macOS desktop toolkit is
replaced by the remote variant from ``backend/macarena_native.py``: the same tools, run in the benchmark VM
through the bridge. Used only by ``start_backend.sh`` when ``MACARENA_TRACK=native``; the app never runs this.

    python scripts/macarena/native_backend.py --host 127.0.0.1 --port 18093
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    from backend import macarena_native
    cls = macarena_native.install()
    import tools.navigation.computer as comp
    if comp.MacOSToolkit is not cls or not macarena_native.ACTIVE:
        raise SystemExit("could not install the remote macOS toolkit; refusing to start "
                         "(its tools would act on this computer)")
    print(f"macarena native track: desktop tools run in the VM ({cls.__name__})", flush=True)
    if "--reload" in sys.argv:
        raise SystemExit("--reload starts a fresh interpreter without the remote toolkit; not supported here")
    from backend.__main__ import main as backend_main
    backend_main()


if __name__ == "__main__":
    main()
