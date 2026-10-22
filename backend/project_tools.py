"""Project tools for a mapped folder: git, checks, search, background processes.

Git writes go through the same hardened git as checkpoints (no repo
config, no filters, no hooks). They do not run inside the shell sandbox,
because that sandbox denies ``.git``. The action policy still asks
before a commit, branch, worktree, or pull request.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_BRANCH = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,80}")
_CODE_SUFFIXES = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb",
    ".md", ".toml", ".yml", ".yaml", ".json", ".sh", ".css", ".html",
}
_CHECK_TIMEOUT = 45


def _workspace(session_id: str):
    from backend.session_manager import _read_workspace_meta
    workspace = _read_workspace_meta(session_id)
    if workspace is None or not workspace.grant_id:
        return None
    return workspace


def _root(session_id: str) -> Path | None:
    workspace = _workspace(session_id)
    if workspace is None:
        return None
    return Path(workspace.host_path)


def commit_worktree(root: Path, message: str) -> dict[str, Any]:
    """Commit the working tree without running filters or hooks."""
    from backend.workspace_git import _git, _index_tree, repo_root

    text = (message or "").strip()
    if not text:
        return {"ok": False, "error": "A commit message is required."}
    repo = repo_root(root)
    if repo is None:
        return {"ok": False, "error": "This folder is not a git repository."}
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode != 0:
        return {"ok": False, "error": "There is no commit yet. Create the first one outside Otto."}
    tree, err = _index_tree(repo)
    if not tree:
        return {"ok": False, "error": err or "Could not hash the working tree."}
    parent = head.stdout.strip()
    made = _git(repo, "commit-tree", tree, "-p", parent, "-F", "-", stdin=text.encode("utf-8"))
    if made.returncode != 0:
        return {"ok": False, "error": (made.stderr or "commit-tree failed").strip()[:400]}
    sha = made.stdout.strip()
    updated = _git(repo, "update-ref", "HEAD", sha, parent)
    if updated.returncode != 0:
        return {"ok": False, "error": (updated.stderr or "update-ref failed").strip()[:400]}
    return {"ok": True, "sha": sha}


def create_branch(root: Path, name: str) -> dict[str, Any]:
    from backend.workspace_git import _git, repo_root

    if not _BRANCH.fullmatch(name or "") or ".." in name:
        return {"ok": False, "error": "That branch name is not allowed."}
    repo = repo_root(root)
    if repo is None:
        return {"ok": False, "error": "This folder is not a git repository."}
    made = _git(repo, "branch", "--", name)
    if made.returncode != 0:
        return {"ok": False, "error": (made.stderr or "branch failed").strip()[:400]}
    return {"ok": True, "branch": name}


def add_worktree(root: Path, name: str) -> dict[str, Any]:
    from backend.workspace_git import _git, repo_root

    if not _BRANCH.fullmatch(name or "") or ".." in name or "/" in name:
        return {"ok": False, "error": "That worktree name is not allowed."}
    repo = repo_root(root)
    if repo is None:
        return {"ok": False, "error": "This folder is not a git repository."}
    dest = repo / ".otto" / "worktrees" / name
    if dest.exists():
        return {"ok": False, "error": "That worktree already exists."}
    dest.parent.mkdir(parents=True, exist_ok=True)
    made = _git(repo, "worktree", "add", "-b", name, str(dest))
    if made.returncode != 0:
        return {"ok": False, "error": (made.stderr or "worktree add failed").strip()[:400]}
    return {"ok": True, "path": str(dest), "branch": name}


def create_pull_request(root: Path, title: str, body: str) -> dict[str, Any]:
    from backend.workspace_git import _git_env, repo_root

    title = (title or "").strip()
    if not title:
        return {"ok": False, "error": "A pull request title is required."}
    repo = repo_root(root)
    if repo is None:
        return {"ok": False, "error": "This folder is not a git repository."}
    env = _git_env()
    env["PATH"] = os.environ.get("PATH", "/usr/bin:/bin")
    try:
        proc = subprocess.run(
            ["gh", "pr", "create", "--title", title, "--body", body or ""],
            cwd=str(repo),
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "gh is not installed."}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "gh pr create timed out."}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "gh failed").strip()[:500]}
    return {"ok": True, "url": (proc.stdout or "").strip()}


def detect_check_commands(root: Path) -> list[str]:
    """The project's own typecheck and lint commands, if it declares them."""
    commands: list[str] = []
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts") or {}
        except (OSError, json.JSONDecodeError):
            scripts = {}
        if isinstance(scripts, dict):
            if "typecheck" in scripts:
                commands.append("npm run typecheck")
            elif (root / "tsconfig.json").is_file():
                commands.append("npx tsc --noEmit --pretty false")
            if "lint" in scripts:
                commands.append("npm run lint")
    if (root / "pyproject.toml").is_file() or (root / "ruff.toml").is_file():
        commands.append("ruff check .")
    return commands[:2]


def auto_project_check(session_id: str) -> str:
    """Run one declared check when this turn wrote project files."""
    from backend.workspace_git import _agent_paths, _latest_turn, _load_state

    root = _root(session_id)
    if root is None or not detect_check_commands(root):
        return ""
    try:
        state = _load_state(session_id)
        if not _agent_paths(state, _latest_turn(state)):
            return ""
    except Exception:
        return ""
    text = run_project_checks(root, timeout=20)
    if not text:
        return ""
    return "Project check after this turn:\n\n" + text


def run_project_checks(root: Path, timeout: int = _CHECK_TIMEOUT) -> str:
    """Run the detected checks. Empty when the project declares none."""
    commands = detect_check_commands(root)
    if not commands:
        return ""
    from backend.exec_sandbox import run_shell, sandbox_settings
    from backend.shell_policy import shell_network
    from backend.workspace_grants import find_by_realpath

    enabled, global_network, _read, _write = sandbox_settings()
    grant = find_by_realpath(str(root))
    network, hosts = shell_network(
        global_network,
        bool(grant and grant.network),
        list(grant.network_hosts) if grant else [],
    )
    parts: list[str] = []
    for command in commands:
        result = run_shell(
            command,
            cwd=root,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "")},
            timeout=timeout,
            sandbox=enabled,
            network=network,
            network_hosts=hosts,
            write_paths=[root],
        )
        text = result.error or "\n".join(piece for piece in (result.stdout, result.stderr) if piece)
        parts.append(f"$ {command}\n{(text or '').strip()[:4000]}\nexit {result.returncode}")
    return "\n\n".join(parts)


def grep_project(root: Path, query: str, limit: int = 30) -> list[str]:
    from backend.workspace import skip_dir_name

    needle = (query or "").strip()
    if not needle:
        return []
    hits: list[str] = []
    lowered = needle.lower()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not skip_dir_name(name)]
        for name in filenames:
            if len(hits) >= limit:
                return hits
            path = Path(dirpath) / name
            if path.suffix.lower() not in _CODE_SUFFIXES and name not in {"Makefile", "Dockerfile"}:
                continue
            try:
                if path.stat().st_size > 200_000:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            for number, line in enumerate(text.splitlines(), 1):
                if lowered in line.lower():
                    hits.append(f"{rel}:{number}:{line.strip()[:200]}")
                    break
    return hits


def _shell_context(session_id: str) -> tuple[Path, bool, bool, list[str]] | None:
    """``(cwd, sandbox, network, hosts)`` for a background process."""
    root = _root(session_id)
    if root is None:
        return None
    from backend.exec_sandbox import sandbox_settings
    from backend.shell_policy import shell_network
    from backend.workspace_grants import get_grant

    workspace = _workspace(session_id)
    grant = get_grant(workspace.grant_id) if workspace is not None else None
    enabled, global_network, _read, _write = sandbox_settings()
    network, hosts = shell_network(
        global_network,
        bool(grant and grant.network),
        list(grant.network_hosts) if grant else [],
    )
    return root, enabled, network, hosts


def build_project_tools(session_id: str) -> list:
    """Tools that only make sense once a folder is mapped."""

    @tool
    def git_commit(message: str) -> str:
        """Commit the mapped project's working tree. Ask the user before calling this.

        The commit is made without git filters or hooks. It includes every
        change in the working tree, including edits the user made outside Otto.
        """
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        result = commit_worktree(root, message)
        if not result.get("ok"):
            return str(result.get("error") or "Commit failed.")
        return f"Committed {result['sha']}"

    @tool
    def git_branch(name: str) -> str:
        """Create a branch in the mapped project. Ask the user before calling this."""
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        result = create_branch(root, name)
        if not result.get("ok"):
            return str(result.get("error") or "Branch failed.")
        return f"Created branch {result['branch']}"

    @tool
    def git_worktree(name: str) -> str:
        """Add a git worktree under .otto/worktrees for parallel work. Ask first."""
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        result = add_worktree(root, name)
        if not result.get("ok"):
            return str(result.get("error") or "Worktree failed.")
        return f"Worktree {result['path']} on {result['branch']}"

    @tool
    def git_pr(title: str, body: str = "") -> str:
        """Open a pull request with gh for the mapped project. Ask the user first.

        Requires the gh CLI and a logged-in account. Network must be allowed
        for the machine. Otto does not push; the branch has to be on the remote.
        """
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        result = create_pull_request(root, title, body)
        if not result.get("ok"):
            return str(result.get("error") or "Pull request failed.")
        return result.get("url") or "Pull request created."

    @tool
    def project_check() -> str:
        """Run the mapped project's typecheck and lint, if it declares them.

        Call this after edits. It uses package.json scripts named typecheck
        or lint, or ruff when the project has a ruff config.
        """
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        text = run_project_checks(root)
        return text or "This project does not declare a typecheck or lint command."

    @tool
    def search_code(query: str) -> str:
        """Search the mapped project by text, then by the local embedding index when files are indexed."""
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        hits = grep_project(root, query)
        lines = ["Text matches:"]
        lines.extend(hits or ["(none)"])
        semantic = _semantic_hits(root, query)
        if semantic:
            lines.append("")
            lines.append("Indexed matches:")
            lines.extend(semantic)
        return "\n".join(lines)

    @tool
    def index_project() -> str:
        """Index up to 80 source files from the mapped project into Otto's embedding index.

        Later search_code calls can return meaning matches. Needs Memory embeddings enabled.
        """
        root = _root(session_id)
        if root is None:
            return "No project folder is mapped."
        return _index_project(root)

    @tool
    def process_start(command: str) -> str:
        """Start a long-running command in the mapped project (a dev server or watcher).

        Returns an id. Read it with process_output, stop it with process_stop.
        The user sees it live in the Terminal tab. The shell approval card still applies.
        """
        if _root(session_id) is None:
            return "No project folder is mapped."
        from backend.background_proc import start
        from backend.terminal_context import session_mode
        mode = session_mode(session_id)
        if mode in ("plan", "ask"):
            return f"{mode} mode cannot start a background process."
        result = start(session_id, command)
        if not result.get("ok"):
            return str(result.get("error") or "Could not start the process.")
        return f"Started {result['id']} (pid {result.get('pid')}). Shown live in the Terminal tab."

    @tool
    def process_output(proc_id: str) -> str:
        """Read the output of a background process started in this session."""
        from backend.background_proc import output
        result = output(session_id, proc_id)
        if not result.get("ok"):
            return str(result.get("error"))
        state = "running" if result.get("running") else f"exited {result.get('exit_code')}"
        return f"{proc_id} {state}\n{result.get('output') or ''}"

    @tool
    def process_stop(proc_id: str) -> str:
        """Stop a background process started in this session."""
        from backend.background_proc import stop
        result = stop(session_id, proc_id)
        return "Stopped." if result.get("ok") else str(result.get("error"))

    @tool
    def process_ports(proc_id: str) -> str:
        """List TCP ports a background process is listening on."""
        from backend.background_proc import ports
        result = ports(session_id, proc_id)
        if not result.get("ok"):
            return str(result.get("error"))
        found = result.get("ports") or []
        note = result.get("note") or ""
        if not found:
            return note or "No listening TCP ports."
        return "Listening: " + ", ".join(str(port) for port in found)

    @tool
    def terminal_run(command: str, terminal_id: str = "", timeout_seconds: int = 120) -> str:
        """Run a command in the user's Terminal tab and return its output and exit code.

        Uses the first open shell terminal (opening one if none exists), so the
        user watches it run and the shell keeps its directory and variables
        between calls. Same sandbox and approval card as execute. Use
        process_start for a server that must keep running.
        """
        from backend import terminal
        from backend.terminal_context import open_terminal, session_mode
        if _root(session_id) is None:
            return "No project folder is mapped."
        if session_mode(session_id) in ("plan", "ask"):
            return "Plan and Ask modes cannot use the terminal. Use execute for read-only commands."
        term = terminal.get(session_id, terminal_id) if terminal_id else None
        if term is None:
            term = next((t for t in terminal.list_terminals(session_id) if t.kind == "shell" and t.alive), None)
        if term is None:
            opened = open_terminal(session_id, kind="shell")
            if not opened.get("ok"):
                return str(opened.get("error") or "Could not open a terminal.")
            term = opened["terminal"]
            import time
            time.sleep(1.0)
        result = terminal.run_in_terminal(term, command, timeout=max(5, min(int(timeout_seconds), 600)))
        if not result.get("ok"):
            return str(result.get("error"))
        text = result.get("output") or "<no output>"
        if len(text) > 12000:
            text = "... (earlier output cut)\n" + text[-12000:]
        if result.get("exit_code") is None:
            return f"[terminal {term.id}] still running after {timeout_seconds}s. Read more with terminal_read.\n{text}"
        suffix = "" if result["exit_code"] == 0 else f"\n\nExit code: {result['exit_code']}"
        return f"[terminal {term.id}]\n{text}{suffix}"

    @tool
    def terminal_read(terminal_id: str = "", lines: int = 80) -> str:
        """Read the latest lines shown in a terminal (the user's shell or a process).

        With no id, reads the most recently opened terminal. Use it to see what
        the user ran or to check a server for new errors.
        """
        from backend import terminal
        terms = terminal.list_terminals(session_id)
        term = terminal.get(session_id, terminal_id) if terminal_id else (terms[-1] if terms else None)
        if term is None:
            return "No terminal is open."
        text, _ = term.text_since(0)
        tail = terminal.strip_ansi(text).rstrip().split("\n")[-max(1, min(int(lines), 400)):]
        state = "running" if term.alive else f"exited {term.exit_code}"
        return f"[terminal {term.id} {term.kind} {state}]\n" + "\n".join(tail)

    @tool
    def run_tests(command: str = "", timeout_seconds: int = 300) -> str:
        """Run the project's tests and return a short result: pass or fail, counts, failing test names.

        With no command it detects pytest, npm test, go test, cargo test, or
        make test. Sandboxed like execute. Read the failure output before you edit.
        """
        from backend.exec_sandbox import passthrough_names, run_shell, scrub_env
        from backend.output_analysis import detect_test_command, summarize_tests
        from backend.terminal_context import build_context
        ctx, reason = build_context(session_id)
        if ctx is None:
            return reason or "No project folder is mapped."
        chosen = (command or "").strip() or detect_test_command(ctx.cwd)
        if not chosen:
            return "No test command found. Pass one, for example run_tests('pytest -q')."
        result = run_shell(
            chosen,
            cwd=ctx.cwd,
            env=scrub_env(os.environ, extra=ctx.env, passthrough=passthrough_names()),
            timeout=max(10, min(int(timeout_seconds), 900)),
            sandbox=ctx.sandbox,
            network=ctx.network,
            network_hosts=ctx.network_hosts,
            write_paths=ctx.write_paths,
            read_paths=ctx.read_paths,
        )
        if result.error:
            return result.error
        text = "\n".join(piece for piece in (result.stdout, result.stderr) if piece)
        report = summarize_tests(text, result.returncode)
        lines = [f"$ {chosen}", f"{report['status'].upper()}: {report['summary']}"]
        if report["failures"]:
            lines.append("Failing: " + ", ".join(report["failures"]))
        if report["status"] != "passed":
            lines.append("--- output (tail) ---")
            lines.append(text.strip()[-5000:])
        return "\n".join(lines)

    @tool
    def check_app(command: str, wait_seconds: int = 30) -> str:
        """Start an app or dev server, wait for it to listen, and report startup errors.

        Returns the process id, the port it opened (if any), and error blocks
        from its log. If the same command is already running it is checked
        instead of started twice. For a web app, open the port with the browser
        tool next.
        """
        import time
        from backend import terminal
        from backend.output_analysis import extract_errors
        from backend.terminal_context import session_mode
        if _root(session_id) is None:
            return "No project folder is mapped."
        if session_mode(session_id) in ("plan", "ask"):
            return "Plan and Ask modes cannot start an app."
        term = next(
            (t for t in terminal.list_terminals(session_id)
             if t.kind == "process" and t.alive and (t.command or "").strip() == command.strip()),
            None,
        )
        if term is None:
            from backend.background_proc import start
            started = start(session_id, command)
            if not started.get("ok"):
                return str(started.get("error") or "Could not start the app.")
            term = terminal.get(session_id, started["id"])
        deadline = time.time() + max(3, min(int(wait_seconds), 120))
        found: list[int] = []
        while time.time() < deadline:
            if term is None or not term.alive:
                break
            found = terminal.listening_ports(term)
            if found:
                time.sleep(1.0)  # let a first request's error reach the log
                break
            time.sleep(1.0)
        text, _ = term.text_since(0) if term else ("", 0)
        errors = extract_errors(text)
        if term is not None and term.alive:
            state = f"running, listening on {', '.join(str(p) for p in found)}" if found else "running, no port opened yet"
        else:
            state = f"exited with code {term.exit_code if term else '?'}"
        lines = [f"[process {term.id if term else '-'}] {state}"]
        if errors:
            lines.append("Errors in the log:")
            lines.extend(f"---\n{block}" for block in errors)
        else:
            lines.append("No errors in the log.")
        return "\n".join(lines)

    return [
        git_commit, git_branch, git_worktree, git_pr, project_check,
        search_code, index_project, process_start, process_output,
        process_stop, process_ports, terminal_run, terminal_read,
        run_tests, check_app,
    ]


def _run_async(coro):
    import asyncio
    import concurrent.futures

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _semantic_hits(root: Path, query: str) -> list[str]:
    try:
        from backend.config import AppConfig
        if not AppConfig.load().memory.embedding_enabled:
            return []
        from backend.embedding_index import get_embedding_index
        index = _run_async(get_embedding_index())
        results = _run_async(index.search(query, k=8, source_type="file"))
    except RuntimeError:
        return []
    except Exception:
        logger.debug("code index search failed", exc_info=True)
        return []
    prefix = str(root.resolve())
    lines: list[str] = []
    for row in results:
        path = str(row.get("source_path") or "")
        if not path.startswith(prefix):
            continue
        lines.append(f"{path}: {str(row.get('text') or '').strip()[:180]}")
    return lines


def _index_project(root: Path) -> str:
    try:
        from backend.config import AppConfig
        if not AppConfig.load().memory.embedding_enabled:
            return "Embeddings are off. Enable Memory in Settings to index this project."
        from backend.embedding_index import get_embedding_index
        from backend.workspace import skip_dir_name
        index = _run_async(get_embedding_index())
    except Exception as exc:
        return f"Could not open the embedding index: {exc}"
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not skip_dir_name(name)]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in _CODE_SUFFIXES:
                files.append(path)
            if len(files) >= 80:
                break
        if len(files) >= 80:
            break
    indexed = 0
    for path in files:
        try:
            indexed += int(_run_async(index.index_file(path, source_type="file")) or 0)
        except Exception:
            logger.debug("index file failed for %s", path, exc_info=True)
    return f"Indexed {indexed} chunks from {len(files)} files."
