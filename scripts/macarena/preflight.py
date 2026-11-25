"""Find out, before a run, which tasks cannot work in the VM because of a missing app or permission.

A MacArena task sets itself up and is graded by ``osascript`` commands that the in-VM server runs. macOS lets a
process script another app only after the user has allowed it (Automation, plus Contacts / Reminders / Calendar
access for the data apps). Nobody clicks Allow in a benchmark VM, so without a grant the command fails with
``-1743`` (not authorized). MacArena starts those setup commands without waiting for them, so the failure is
silent: the contact the task should edit is never created, and the grader (which has the same permission
problem) says False whatever the agent did. These tasks would score 0.0 for a reason that has nothing to do with
the agent, so the runner probes each scripted app once and leaves those tasks out.

Stdlib only: the runner and the bridge both import it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

# ``tell application "X"`` in a task file, whether the quotes are plain or escaped inside a JSON string.
_TELL_APP = re.compile(r'tell application (?:id )?\\*"([^"\\]+)')

NOT_AUTHORIZED = ("-1743", "not authorized")
NOT_INSTALLED = ("-10814", "can’t get application", "can't get application", "unable to find application",
                 "application isn’t running", "no application knows")

FIX_HINT = (
    "Answer the permission dialogs once in the base VM image: run `scripts/macarena/bench.sh grant-apps` (it boots "
    "the base VM, triggers the dialogs and waits while you click Allow). Until then these tasks are skipped, "
    "because they would score 0.0 whatever the agent does. Use --no-preflight to run them anyway."
)


def apps_in_task(path: str | Path) -> set[str]:
    """Apps a task's setup or grading script talks to with AppleScript."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return set()
    return {m.group(1).strip() for m in _TELL_APP.finditer(text) if m.group(1).strip()}


def needed_apps(tasks: Iterable[dict]) -> set[str]:
    out: set[str] = set()
    for t in tasks:
        out |= apps_in_task(t["path"])
    return out


# What to ask a data app. Its name is not enough: Contacts and Reminders answer "get name" without the data
# permission, and then hang forever on the first real query, waiting for a dialog nobody clicks.
DATA_QUERY = {"Contacts": "count of people", "Reminders": "count of lists", "Calendar": "count of calendars",
              "Notes": "count of folders", "System Events": "name of login items"}


def probe_command(app: str) -> str:
    """Shell command that asks ``app`` a question its tasks' scripts also ask. It starts the app and needs the
    permission."""
    safe = app.replace('"', "").replace("'", "").replace("\\", "")
    what = DATA_QUERY.get(safe, "get name")
    verb = what if what.startswith("get ") else f"get {what}"
    return f"osascript -e 'tell application \"{safe}\" to {verb}' 2>&1"


def classify(returncode, text: str) -> dict:
    low = (text or "").lower()
    if "timed out" in low:
        return {"status": "prompt_pending",
                "detail": "no answer: a permission dialog is probably waiting for a click"}
    if any(k in low for k in NOT_AUTHORIZED):
        return {"status": "not_authorized", "detail": (text or "").strip()[:160]}
    if any(k in low for k in NOT_INSTALLED):
        return {"status": "not_installed", "detail": (text or "").strip()[:160]}
    if returncode in (0, None) and "error" not in low:
        return {"status": "ok", "detail": (text or "").strip()[:80]}
    return {"status": "error", "detail": (text or "").strip()[:160] or f"exit code {returncode}"}


def split_tasks(tasks: list[dict], results: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """(tasks that can run, tasks left out with the reason)."""
    runnable, blocked = [], []
    for t in tasks:
        bad = {a: results[a] for a in apps_in_task(t["path"]) if a in results and results[a]["status"] != "ok"}
        if bad:
            blocked.append({"id": t["id"], "domain": t["domain"],
                            "reason": "; ".join(f"{a}: {r['status']}" for a, r in sorted(bad.items()))})
        else:
            runnable.append(t)
    return runnable, blocked


def render(results: dict[str, dict]) -> str:
    lines = ["app permission check (what the setup and grading scripts need):"]
    for app in sorted(results):
        r = results[app]
        lines.append(f"  {app:<22} {r['status']:<15} {r['detail'] if r['status'] != 'ok' else ''}".rstrip())
    return "\n".join(lines)


def dump(results: dict) -> str:
    return json.dumps(results, indent=1, sort_keys=True)


def _main(argv: list[str]) -> int:
    """``python preflight.py apps DIR...``: the apps the task files under each DIR script with AppleScript, one per
    line. ``bench.sh verify-image`` uses it to know what to probe in a VM image."""
    if len(argv) < 2 or argv[0] != "apps":
        print("usage: preflight.py apps DIR [DIR ...]")
        return 2
    found: set[str] = set()
    for d in argv[1:]:
        for p in Path(d).rglob("*.json"):
            found |= apps_in_task(p)
    print("\n".join(sorted(found)))
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
