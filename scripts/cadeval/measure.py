"""Runs in the CAD Tools Python (it has cadquery): print the metrics of a program's `result` or of a STEP file
as one JSON line, using the same code cad_run uses, so the eval and the tool measure alike.

    python measure.py code part.py        python measure.py step part.step
"""
import json
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend" / "builtin_mcps" / "cad_tools"))
import _probe as P  # noqa: E402


def main() -> None:
    mode, path = sys.argv[1], sys.argv[2]
    try:
        if mode == "code":
            os.chdir(os.path.dirname(os.path.abspath(path)) or ".")
            ns = runpy.run_path(os.path.abspath(path), run_name="__main__")
            sh = P._shape(ns["result"])
        else:
            import cadquery as cq
            sh = P._shape(cq.importers.importStep(path))
        print(json.dumps({"ok": True, "metrics": P._metrics(sh)}))
    except BaseException as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}))


if __name__ == "__main__":
    main()
