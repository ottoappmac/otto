"""Entry point for running the backend server directly.

Usage:
    python -m backend                      # default: host=0.0.0.0, port=18081
    python -m backend --port 9000          # custom port
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Otto backend server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=18081, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    # Ensure src/ is on the path so deep_agent, tools, etc. resolve
    from pathlib import Path
    src_dir = str(Path(__file__).resolve().parent.parent / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    import uvicorn
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent

    reload_kwargs: dict = {}
    if args.reload:
        # Scope the watcher to just the Python source directories. This keeps
        # the Tauri build output (app/src-tauri/target/) and the venv out of
        # the watch set without needing reload_excludes — those globs made
        # uvicorn walk those huge trees at startup and hang before binding.
        reload_kwargs = {
            "reload": True,
            "reload_dirs": [
                str(root / "backend"),
                str(root / "src"),
            ],
        }

    uvicorn.run(
        "backend.server:app",
        host=args.host,
        port=args.port,
        **reload_kwargs,
    )


if __name__ == "__main__":
    main()
