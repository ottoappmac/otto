"""Run SWE-bench's evaluation harness with the offline-safe GitHub cache installed.

Takes exactly the arguments of ``python -m swebench.harness.run_evaluation``.
"""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import http_cache  # noqa: E402

http_cache.install()
runpy.run_module("swebench.harness.run_evaluation", run_name="__main__", alter_sys=True)
