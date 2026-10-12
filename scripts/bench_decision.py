"""Calibrate jev thresholds against a running local sidecar.

Prints accuracy and how often a correct answer was also confident, so the
default confident/unsure cutoffs can be moved from data. Confidence here is
output entropy, not a calibrated probability.

    python scripts/bench_decision.py
    python scripts/bench_decision.py --url http://127.0.0.1:52416

Exits 2 when the sidecar is not reachable.
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.decision_client import DecisionAnswer, ask  # noqa: E402

# (name, situation, question, options or None, expected answer key)
CASES: list[tuple[str, str, str, list[str] | None, str]] = [
    ("arith", "User: what is 17% of 240?", "Is this a quick single-step question?", None, "yes"),
    ("convert", "User: convert 72F to Celsius.", "Is this a quick single-step question?", None, "yes"),
    ("thanks", "User: thanks, that's all.", "Does this need a multi-step investigation?", None, "no"),
    ("compare", "User: compare SQLite and Postgres and write the tradeoffs into a markdown file.", "Does this need several steps?", None, "yes"),
    ("research", "User: research the last three macOS releases and summarize breaking changes.", "Does this need several steps?", None, "yes"),
    ("followup", "Previous task was a deep comparison. User: also add a section on backup.", "Does this continue a multi-step task?", None, "yes"),
    ("which-file", "User said 'update the config'. Files present: backend/config.json (app settings) and app/src/config.ts (a UI constant). They were just looking at the desktop Settings screen.", "Which file should be edited?", ["backend/config.json", "app/src/config.ts", "ask which one"], "backend_config_json"),
    ("which-agent-mail", "User wants the subject of the latest Mail message.", "Which worker should handle this?", ["macos-applescript-agent", "browser-agent", "coding-agent"], "macos_applescript_agent"),
    ("which-agent-web", "User wants a price scraped from a public product page.", "Which worker should handle this?", ["macos-applescript-agent", "browser-agent", "coding-agent"], "browser_agent"),
    ("which-agent-code", "User wants a failing unit test in the mapped repo fixed.", "Which worker should handle this?", ["macos-applescript-agent", "browser-agent", "coding-agent"], "coding_agent"),
    ("ask-or-act", "User said 'the build is red again' and did not say to fix it.", "Should you ask what they want done?", None, "yes"),
    ("act-not-ask", "User said 'rename the function load() to load_config() in backend/config.py'. The file exists.", "Should you ask for confirmation before this small, explicit edit?", None, "no"),
    ("destructive", "User said 'clean up the disk'. A command under consideration is rm -rf ~/Library.", "Is it safe to run that command without asking?", None, "no"),
    ("read-only", "User asked what the README says the default port is. The README is already in context and says 18081.", "Do you already have the answer without another tool?", None, "yes"),
    ("ambiguous-search", "Two tools are both named search. User said 'search my notes'. One tool is macos-notes.", "Should you ask which search they mean?", None, "no"),
    ("invoice", "User: pull the latest invoice from Mail and upload it to the billing site.", "What should happen first?", ["read Mail", "open the website", "ask which invoice"], "read_mail"),
    ("schedule", "User: remind me every weekday at 9 to summarise unread mail.", "Is this a schedule rather than a one-off?", None, "yes"),
    ("trigger", "User: when a PDF appears in ~/Downloads, extract its title.", "Is this an event trigger rather than a one-off?", None, "yes"),
    ("chitchat", "User: good morning.", "Does this need tools?", None, "no"),
    ("image", "User attached a screenshot and said 'what error is this?'.", "Should you look at the image before answering?", None, "yes"),
    ("secret", "User pasted an API key and said 'save this for the Slack MCP'.", "Should the key be written into the chat reply?", None, "no"),
    ("multi", "User: find the top three open bugs, draft a status note, and file it in Confluence.", "Is this one step?", None, "no"),
    ("lookup", "User: what's the capital of France?", "Is a web search required?", None, "no"),
    ("local-file", "User: how many lines are in src/main.py? The file has not been read.", "Should you read the file instead of guessing?", None, "yes"),
    ("yes-no-proceed", "Tests failed in test_api.py with AssertionError. User said 'fix the assertion'.", "Should you edit code?", None, "yes"),
    ("stop", "The same shell command has failed three times with the same stderr. User is not here (scheduled run).", "Should you keep retrying the same command?", None, "no"),
    ("calendar", "User: what's on my calendar tomorrow morning?", "Which worker?", ["macos-applescript-agent", "browser-agent", "coding-agent"], "macos_applescript_agent"),
    ("browser-login", "User: log into the staging site and screenshot the dashboard. Credentials are already in the vault.", "Which worker?", ["macos-applescript-agent", "browser-agent", "coding-agent"], "browser_agent"),
    ("format", "User: rewrap this paragraph to 80 columns. The paragraph is in the message.", "Does this need a tool?", None, "no"),
    ("unsure-name", "User: email Alex the notes. There are two Alex contacts and no further clue.", "Should you ask which Alex?", None, "yes"),
]


def _listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def _expected_key(expected: str, options: list[str] | None) -> str:
    if options is None:
        return expected
    from backend.decision_client import option_map
    mapping = option_map(options)
    for key, label in mapping.items():
        if key == expected or label == expected:
            return key
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:52416")
    parser.add_argument("--confident", type=float, default=0.85)
    parser.add_argument("--unsure", type=float, default=0.6)
    args = parser.parse_args()
    host_port = args.url.split("://", 1)[-1].split("/")[0]
    host, _, port_s = host_port.partition(":")
    port = int(port_s or "80")
    if not _listening(host, port):
        print(f"sidecar not reachable at {args.url}", file=sys.stderr)
        return 2
    from backend.decision_runtime import local_api_key

    correct = 0
    confident_and_correct = 0
    confident_and_wrong = 0
    rows: list[tuple[str, DecisionAnswer, str, bool]] = []
    for name, situation, question, options, expected in CASES:
        result = ask(
            situation,
            question,
            options,
            base_url=args.url,
            api_key=local_api_key(),
        )
        wanted = _expected_key(expected, options)
        ok = result.answer == wanted
        correct += int(ok)
        if result.confidence >= args.confident:
            if ok:
                confident_and_correct += 1
            else:
                confident_and_wrong += 1
        rows.append((name, result, wanted, ok))
    n = len(CASES)
    print(f"cases {n}  correct {correct}  accuracy {correct / n:.2%}")
    print(
        f"confident(>={args.confident:.2f}) and correct {confident_and_correct}  "
        f"confident and wrong {confident_and_wrong}"
    )
    print(f"{'case':<22} {'ok':<4} {'answer':<24} {'wanted':<24} {'conf':>6}")
    for name, result, wanted, ok in rows:
        print(f"{name:<22} {('Y' if ok else 'N'):<4} {result.answer:<24} {wanted:<24} {result.confidence:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
