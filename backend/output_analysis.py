"""Read test and server output so the agent gets a short, structured result."""

from __future__ import annotations

import json
import re
from pathlib import Path

from backend.terminal import strip_ansi

_PYTEST_SUMMARY = re.compile(r"=+ (?P<body>.*?(?:passed|failed|error|no tests ran).*?) in [\d.]+s", re.I)
_PYTEST_FAILED = re.compile(r"^(?:FAILED|ERROR) (\S+)", re.M)
_JEST_SUMMARY = re.compile(r"^Tests:\s+(?P<body>.+)$", re.M)
_JEST_FAILED = re.compile(r"^\s*●\s+(.+?)$", re.M)
_GO_FAILED = re.compile(r"^--- FAIL: (\S+)", re.M)
_GO_OK = re.compile(r"^ok\s+\S+", re.M)
_CARGO_SUMMARY = re.compile(r"^test result: (?P<body>.+)$", re.M)
_CARGO_FAILED = re.compile(r"^test (\S+) \.\.\. FAILED", re.M)
_NODE_TEST_SUMMARY = re.compile(r"^ℹ (pass|fail) (\d+)", re.M)

_ERROR_LINE = re.compile(
    r"(Traceback \(most recent call last\)|\bError:|\bException\b|ERR!|\bFATAL\b|\bpanic:|"
    r"EADDRINUSE|ModuleNotFoundError|ImportError|SyntaxError|Cannot find module|"
    r"Failed to compile|compilation failed|address already in use|command not found|"
    r"Operation not permitted)",
    re.I,
)


def detect_dev_command(root: Path) -> str:
    """The project's own dev-server command, or an empty string."""
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts") or {}
        except (OSError, json.JSONDecodeError):
            scripts = {}
        if isinstance(scripts, dict):
            for name in ("dev", "start", "serve"):
                if name in scripts:
                    return f"npm run {name}"
    if (root / "manage.py").is_file():
        return "python3 manage.py runserver"
    makefile = root / "Makefile"
    if makefile.is_file():
        try:
            target = re.search(r"^(dev|serve|run):", makefile.read_text(encoding="utf-8"), re.M)
        except OSError:
            target = None
        if target:
            return f"make {target.group(1)}"
    return ""


def detect_test_command(root: Path) -> str:
    """The project's own test command, or an empty string."""
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts") or {}
        except (OSError, json.JSONDecodeError):
            scripts = {}
        if isinstance(scripts, dict) and "test" in scripts:
            return "npm test --silent"
    if (root / "go.mod").is_file():
        return "go test ./..."
    if (root / "Cargo.toml").is_file():
        return "cargo test"
    if (root / "pytest.ini").is_file() or (root / "pyproject.toml").is_file() or (root / "tests").is_dir():
        return "python3 -m pytest -q"
    if (root / "Makefile").is_file():
        try:
            if re.search(r"^test:", (root / "Makefile").read_text(encoding="utf-8"), re.M):
                return "make test"
        except OSError:
            pass
    return ""


def summarize_tests(output: str, exit_code: int) -> dict:
    """``{status, summary, failures}`` from common runners. Falls back to the exit code."""
    text = strip_ansi(output)
    summary = ""
    failures: list[str] = []
    match = _PYTEST_SUMMARY.search(text)
    if match:
        summary = match.group("body").strip()
        failures = _PYTEST_FAILED.findall(text)
    else:
        match = _JEST_SUMMARY.search(text)
        if match:
            summary = match.group("body").strip()
            failures = list(dict.fromkeys(_JEST_FAILED.findall(text)))
        elif _CARGO_SUMMARY.search(text):
            summary = "; ".join(m.group("body").strip() for m in _CARGO_SUMMARY.finditer(text))
            failures = _CARGO_FAILED.findall(text)
        elif _GO_FAILED.search(text) or _GO_OK.search(text):
            failures = _GO_FAILED.findall(text)
            oks = len(_GO_OK.findall(text))
            summary = f"{oks} package(s) ok, {len(failures)} test(s) failed"
        else:
            counts = dict((k, int(v)) for k, v in _NODE_TEST_SUMMARY.findall(text))
            if counts:
                summary = f"{counts.get('pass', 0)} passed, {counts.get('fail', 0)} failed"
    status = "passed" if exit_code == 0 else "failed"
    if not summary:
        summary = "exit code 0" if exit_code == 0 else f"exit code {exit_code}"
    return {"status": status, "summary": summary, "failures": list(dict.fromkeys(failures))[:25]}


def extract_errors(output: str, limit: int = 8, context: int = 2) -> list[str]:
    """Blocks of log lines around error-looking lines, so a stack trace stays whole."""
    lines = strip_ansi(output).split("\n")
    blocks: list[str] = []
    i = 0
    while i < len(lines) and len(blocks) < limit:
        if _ERROR_LINE.search(lines[i]):
            start = max(0, i - 1)
            end = min(len(lines), i + 1 + context)
            # A Python traceback runs until the exception line.
            if "Traceback" in lines[i]:
                end = i + 1
                while end < len(lines) and (lines[end].startswith((" ", "\t")) or end == i + 1):
                    end += 1
                end = min(len(lines), end + 1)
            blocks.append("\n".join(line.rstrip() for line in lines[start:end]).strip())
            i = end
        else:
            i += 1
    return blocks
