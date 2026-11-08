#!/usr/bin/env python
"""Watch what the benchmark agent is doing, live.

    python3 scripts/swebench/watch.py              # follow the newest session
    python3 scripts/swebench/watch.py --all        # every active session, interleaved
    python3 scripts/swebench/watch.py --from-start # replay the newest session, then follow
    python3 scripts/swebench/watch.py --session 31174cec

Reads the session log the benchmark backend writes as it goes (nothing is sent to the
agent). Prints each thought, tool call and a short snippet of each result. Ctrl-C to stop.
Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

DEFAULT_RUNS = Path(os.environ.get("SWEBENCH_RUNS") or Path(__file__).resolve().parents[2].parent / "otto-swebench-runs")
DIM, BOLD, CYAN, GREEN, YELLOW, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"
if not sys.stdout.isatty():
    DIM = BOLD = CYAN = GREEN = YELLOW = RED = RESET = ""


def one_line(text: str, width: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def describe_call(name: str, args: dict, width: int) -> str:
    """The part of a tool call a person wants to read."""
    for key in ("command", "file_path", "path", "pattern", "query", "url"):
        if key in args:
            extra = ""
            if name == "edit_file" and "old_string" in args:
                extra = f"  ({len(str(args.get('old_string','')))}→{len(str(args.get('new_string','')))} chars)"
            if name == "grep" and "path" in args:
                extra = f"  in {args['path']}"
            return one_line(f"{args[key]}{extra}", width)
    if name == "write_todos":
        todos = args.get("todos", [])
        active = next((t["content"] for t in todos if t.get("status") == "in_progress"), "")
        done = sum(1 for t in todos if t.get("status") == "completed")
        return one_line(f"{done}/{len(todos)} done; now: {active}", width)
    return one_line(json.dumps(args), width)


def render(event: dict, tag: str, width: int) -> str | None:
    kind = event.get("type")
    stamp = time.strftime("%H:%M:%S")
    if kind == "agent":
        return f"{DIM}{stamp}{RESET} {tag}{CYAN}says{RESET}  {one_line(event.get('content', ''), width)}"
    if kind == "tool_call":
        meta = event.get("metadata") or {}
        name = event.get("content", "?")
        via = ""
        args = meta.get("args") or {}
        if ".swe/run" in json.dumps(args):
            via = f" {YELLOW}[docker]{RESET}"
        st = meta.get("stats") or {}
        speed = f"  {DIM}{st.get('generation_tokens','?')} tok @ {st.get('generation_tps','?')}/s{RESET}" if st else ""
        return f"{DIM}{stamp}{RESET} {tag}{BOLD}{name}{RESET}{via} {describe_call(name, args, width)}{speed}"
    if kind == "tool_result":
        text = str(event.get("content", ""))
        bad = re.search(r"Traceback|Error|FAILED|failed|exit code [1-9]|exit: [1-9]|No such file", text)
        color = RED if bad else GREEN
        return f"{'':8s} {tag}{color}→{RESET} {DIM}{one_line(text, width)}{RESET}"
    if kind == "user":
        return f"{DIM}{stamp}{RESET} {tag}{BOLD}task sent{RESET}  {one_line(event.get('content', '').split('<issue>')[-1], width)}"
    if kind in ("error", "stopped", "done"):
        return f"{DIM}{stamp}{RESET} {tag}{RED if kind == 'error' else BOLD}{kind.upper()}{RESET} {one_line(event.get('content', ''), width)}"
    return None


def sessions_dir(runs: Path) -> Path:
    return runs / "home" / "Library" / "Application Support" / "Otto" / "sessions"


def newest(d: Path, n: int = 1) -> list[Path]:
    files = sorted(d.glob("*.messages.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default=str(DEFAULT_RUNS))
    p.add_argument("--session", help="session id prefix to follow")
    p.add_argument("--all", action="store_true", help="follow every session active in the last 2 minutes")
    p.add_argument("--from-start", action="store_true", help="replay what already happened first")
    p.add_argument("--width", type=int, default=max(60, os.get_terminal_size().columns - 28) if sys.stdout.isatty() else 160)
    args = p.parse_args()

    d = sessions_dir(Path(args.runs_dir).expanduser())
    if not d.exists():
        sys.exit(f"no sessions folder at {d}. Is the benchmark backend running?")

    offsets: dict[Path, int] = {}
    tags: dict[Path, str] = {}

    def pick() -> list[Path]:
        if args.session:
            return [f for f in d.glob(f"{args.session}*.messages.json")]
        files = newest(d, 6)
        if args.all:
            return [f for f in files if time.time() - f.stat().st_mtime < 120] or files[:1]
        return files[:1]

    print(f"watching {d}  (Ctrl-C to stop)", flush=True)
    try:
        while True:
            for f in pick():
                if f not in offsets:
                    tags[f] = f"{DIM}[{f.name[:6]}]{RESET} " if args.all else ""
                    offsets[f] = 0 if args.from_start else f.stat().st_size
                    print(f"{BOLD}-- following session {f.name[:8]}{RESET}", flush=True)
                with f.open() as fh:
                    fh.seek(offsets[f])
                    chunk = fh.read()
                    complete = chunk.rsplit("\n", 1)[0] if "\n" in chunk else ""
                    offsets[f] += len(complete.encode()) + (1 if complete else 0)
                for line in complete.splitlines():
                    try:
                        out = render(json.loads(line), tags[f], args.width)
                    except json.JSONDecodeError:
                        continue
                    if out:
                        print(out, flush=True)
            time.sleep(1)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
