"""LangChain tools that let the chat agent manage custom triggers."""

from __future__ import annotations

from langchain_core.tools import tool

from backend.run_output import (
    MAX_RUNS_RETURNED as _MAX_RUNS_RETURNED,
    fmt_duration as _fmt_duration,
    list_run_files as _list_run_files,
    read_run_output as _read_run_output,
)
from backend.trigger_manager import MAX_TRIGGERS

_TRIGGER_BUILDER_AGENT = "trigger-builder-agent"

# Trigger types that run arbitrary user-supplied code or make network requests
# on a recurring basis.  Creation is restricted to the dedicated
# trigger-builder-agent so a general-purpose or task agent can't accidentally
# (or maliciously) schedule background network/shell activity.
_PRIVILEGED_TYPES = frozenset({"http", "git", "shell"})


def build_trigger_tools(agent_name: str | None = None) -> list:
    """Build trigger management tools for injection into the agent graph.

    ``agent_name`` is the currently running agent.  When it is NOT the
    ``trigger-builder-agent``, creation of the ``http`` / ``git`` / ``shell``
    trigger types is blocked — those types execute arbitrary network requests
    or shell commands on a recurring schedule and should only be authored by
    the dedicated trigger-builder workflow.

    These tools mirror the schedule tools in shape (list / create /
    update / delete / run-now / toggle) but operate on the polling
    trigger primitive.  The trigger-builder agent uses them to wire
    "fire agent X when condition Y" rules; any agent can use them, but
    the typical caller is the dedicated trigger-builder agent which
    also has agent-management tools so it can create the worker agent
    first when needed.
    """
    # Capture the *caller* name now, before inner tool functions define their
    # own ``agent_name`` parameter which would shadow this outer variable.
    from backend.agent_library import _slugify as _slug
    _caller_agent = _slug(agent_name or "")

    @tool
    def list_triggers() -> str:
        """List all configured triggers with their id, type, agent, poll
        interval, enabled status, and last run info."""
        from backend.trigger_manager import load_all_triggers

        triggers = load_all_triggers()
        if not triggers:
            return "No triggers configured."
        lines = []
        for s in triggers:
            status = "enabled" if s.enabled else "paused"
            last = s.last_run.isoformat() if s.last_run else "never"
            if s.type == "fileos":
                target = f", path={s.path!r}, watch={s.watch}"
            elif s.type == "macostool":
                target = f", language={s.language}"
            elif s.type == "http":
                target = f", url={s.url!r}, mode={s.http_mode}"
            elif s.type == "git":
                target = f", repo={s.repo_path!r}, branch={s.branch}"
            elif s.type == "shell":
                target = f", command={(s.command or '')[:40]!r}, mode={s.shell_mode}"
            else:
                target = ""
            lines.append(
                f"- **{s.id}** ({status}, type={s.type}): "
                f"agent={s.agent_name or 'general-purpose'}, "
                f"poll={s.poll_seconds}s{target}, "
                f"last_run={last}, last_status={s.last_status or 'n/a'}"
            )
        custom_total = sum(1 for t in triggers if not t.builtin)
        builtin_total = sum(1 for t in triggers if t.builtin)
        lines.append(
            f"\n{custom_total}/{MAX_TRIGGERS} custom trigger slots used "
            f"({builtin_total} built-in triggers don't count toward the limit)."
        )
        return "\n".join(lines)

    @tool
    def get_trigger_runs(trigger_id: str, limit: int = 10) -> str:
        """List the run history for a trigger, including each run's outcome.

        Returns, for each past run: run ID, status (success/error/running/
        cancelled), start time, duration, message count, any error, the
        underlying session ID, and a listing of output files the run produced.

        Use this to find the sessions a trigger has spawned and review how it
        has been performing over time. To read a run's actual output, follow
        up with:
          - get_session_messages(session_id) for the agent's conversation/result
          - read_trigger_run_output(trigger_id, run_id, file_path) for a file

        Args:
            trigger_id: ID of the trigger (see list_triggers).
            limit: Max number of most-recent runs to return (1–50, default 10).
        """
        from backend.trigger_manager import load_runs, load_trigger, runs_dir

        if not load_trigger(trigger_id):
            return (
                f"Error: Trigger '{trigger_id}' not found. "
                f"Use list_triggers to see available triggers."
            )

        limit = max(1, min(limit, _MAX_RUNS_RETURNED))
        runs = load_runs(trigger_id, limit=limit)
        if not runs:
            return f"Trigger '{trigger_id}' has no recorded runs yet."

        lines: list[str] = [
            f"{len(runs)} run(s) for trigger '{trigger_id}' (most recent first):\n"
        ]
        for r in runs:
            started = r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else "unknown"
            duration = _fmt_duration(r.started_at, r.finished_at)
            duration_str = f", {duration}" if duration else ""

            files_dir = runs_dir(trigger_id) / r.id / "files"
            file_entries = _list_run_files(files_dir, with_sizes=True)

            lines.append(
                f"- run_id: `{r.id}`\n"
                f"  status: {r.status}{duration_str}\n"
                f"  started: {started}, messages: {r.message_count}\n"
                f"  session_id: {r.session_id or 'n/a'}"
            )
            if r.error:
                lines.append(f"  error: {r.error[:200]}")
            if file_entries:
                shown = file_entries[:10]
                more = f" (+{len(file_entries) - 10} more)" if len(file_entries) > 10 else ""
                lines.append(f"  output_files: {', '.join(shown)}{more}")
            else:
                lines.append("  output_files: (none)")

        return "\n".join(lines)

    @tool
    def read_trigger_run_output(trigger_id: str, run_id: str, file_path: str = "") -> str:
        """Read an output file produced by a specific trigger run.

        Use get_trigger_runs first to find the run_id and the names of the
        output files. Leave file_path empty to list the available files for
        the run.

        Args:
            trigger_id: ID of the trigger.
            run_id: ID of the run (from get_trigger_runs).
            file_path: Relative path of the output file to read. Leave empty
                       to list the files available for this run.
        """
        from backend.trigger_manager import load_trigger, runs_dir
        from backend.utils import is_safe_path_segment

        if not load_trigger(trigger_id):
            return (
                f"Error: Trigger '{trigger_id}' not found. "
                f"Use list_triggers to see available triggers."
            )
        if not is_safe_path_segment(run_id):
            return (
                f"Error: Run '{run_id}' not found. "
                f"Use get_trigger_runs to see available runs."
            )

        files_dir = runs_dir(trigger_id) / run_id / "files"
        return _read_run_output(
            files_dir, run_id, file_path, owner_label=f"trigger '{trigger_id}'"
        )

    @tool
    def create_trigger(
        trigger_id: str,
        type: str,
        prompt: str,
        agent_name: str = "",
        poll_seconds: int = 60,
        # fileos
        path: str = "",
        watch: str = "mtime",
        glob: str = "",
        # macostool
        script: str = "",
        language: str = "AppleScript",
        match: str = "",
        # http
        url: str = "",
        http_mode: str = "body_hash",
        method: str = "GET",
        json_path: str = "",
        # git
        repo_path: str = "",
        branch: str = "HEAD",
        author_filter: str = "",
        path_filter: str = "",
        # shell
        command: str = "",
        shell_mode: str = "stdout_change",
        cwd: str = "",
    ) -> str:
        """Create a new custom trigger.

        Always make sure the target agent already exists BEFORE creating
        the trigger — the trigger fires the agent by name, and a missing
        agent surfaces as a runtime error that's hard to diagnose.  Use
        ``list_existing_agents`` first; if no suitable agent exists,
        call ``create_agent_config`` to build it, THEN create the trigger.

        Args:
            trigger_id: Unique kebab-case identifier (e.g. "downloads-pdf-watch").
            type: ``"fileos"`` | ``"macostool"`` | ``"http"`` | ``"git"`` | ``"shell"``.
            prompt: Instruction the agent receives every time the trigger
                    fires.  Should be self-contained — the trigger appends
                    a JSON event payload describing what happened, so the
                    prompt can reference paths/values via the payload
                    (e.g. "Process the PDF listed in event.new_paths.").
            agent_name: Worker agent that runs when the trigger fires.
                        Leave empty for general-purpose, but providing a
                        focused agent is strongly recommended.
            poll_seconds: How often to check the condition.  5..86400.
                          Default 60.

            path: (fileos) Absolute or ``~``-relative path to watch.
            watch: (fileos) ``"mtime"`` | ``"size"`` | ``"exists"`` |
                   ``"new_files"`` (requires *glob*).
            glob: (fileos, watch=new_files) fnmatch pattern, e.g. ``"*.pdf"``.

            script: (macostool) AppleScript or JXA source code.
            language: (macostool) ``"AppleScript"`` | ``"JavaScript"``.
            match: (macostool/http/shell) Optional regex.  For macostool, gates
                   firing on stdout match.  For http/shell, used in regex mode.

            url: (http) Endpoint URL to poll.
            http_mode: (http) ``"status_change"`` | ``"body_hash"`` |
                       ``"json_value"`` (requires *json_path*) | ``"regex"``
                       (requires *match*).
            method: (http) ``"GET"`` | ``"POST"`` | ``"HEAD"``.
            json_path: (http, json_value mode) Dotted path into JSON response,
                       e.g. ``"data.items.0.id"`` — fires when value changes.

            repo_path: (git) Absolute path to a local git repository.
            branch: (git) Branch to watch (default ``"HEAD"``).
            author_filter: (git) Optional ``--author`` regex.
            path_filter: (git) Optional path glob — only fire if commits
                         touched these paths.

            command: (shell) Shell command to run via ``/bin/sh -c``.
            shell_mode: (shell) ``"stdout_change"`` | ``"regex"`` (requires
                        *match*) | ``"exit_code_change"``.
            cwd: (shell) Working directory for the command (optional).
        """
        from backend.schemas import TriggerSpec
        from backend.trigger_manager import (
            load_all_triggers,
            load_trigger,
            register_job,
            save_trigger,
            validate_spec,
            validate_trigger_id,
        )

        id_err = validate_trigger_id(trigger_id)
        if id_err:
            return f"Error: {id_err}"

        existing = load_all_triggers()
        custom_count = sum(1 for t in existing if not t.builtin)
        if custom_count >= MAX_TRIGGERS:
            return (
                f"Error: Maximum of {MAX_TRIGGERS} custom triggers reached. "
                f"Delete one first with delete_trigger. "
                f"(Built-in triggers don't count toward this limit.)"
            )

        if load_trigger(trigger_id):
            return f"Error: Trigger '{trigger_id}' already exists."

        if type not in ("fileos", "macostool", "http", "git", "shell"):
            return (
                "Error: type must be one of "
                "'fileos' / 'macostool' / 'http' / 'git' / 'shell'."
            )

        if type in _PRIVILEGED_TYPES and _caller_agent != _TRIGGER_BUILDER_AGENT:
            kind = (
                "recurring network requests"
                if type == "http"
                else "git subprocess calls"
                if type == "git"
                else "arbitrary shell commands"
            )
            return (
                f"Error: Creating '{type}' triggers is restricted to the "
                f"trigger-builder-agent because they execute {kind} on a "
                f"recurring schedule. Ask the trigger-builder-agent to create "
                f"this trigger for you, or use 'fileos' / 'macostool' types instead."
            )
        if watch not in ("mtime", "size", "exists", "new_files"):
            return "Error: watch must be one of mtime/size/exists/new_files."
        if language not in ("AppleScript", "JavaScript"):
            return "Error: language must be 'AppleScript' or 'JavaScript'."
        if http_mode not in ("status_change", "body_hash", "json_value", "regex"):
            return (
                "Error: http_mode must be one of "
                "status_change/body_hash/json_value/regex."
            )
        if method not in ("GET", "POST", "HEAD"):
            return "Error: method must be GET / POST / HEAD."
        if shell_mode not in ("stdout_change", "regex", "exit_code_change"):
            return (
                "Error: shell_mode must be one of "
                "stdout_change/regex/exit_code_change."
            )

        # Per-type required-field checks.
        if type == "fileos" and not path:
            return "Error: fileos triggers require 'path'."
        if type == "macostool" and not script:
            return "Error: macostool triggers require 'script'."
        if type == "http":
            if not url:
                return "Error: http triggers require 'url'."
            if http_mode == "json_value" and not json_path:
                return "Error: http_mode='json_value' requires 'json_path'."
            if http_mode == "regex" and not match:
                return "Error: http_mode='regex' requires 'match'."
        if type == "git" and not repo_path:
            return "Error: git triggers require 'repo_path'."
        if type == "shell":
            if not command:
                return "Error: shell triggers require 'command'."
            if shell_mode == "regex" and not match:
                return "Error: shell_mode='regex' requires 'match'."

        if agent_name:
            from backend.agent_library import get_agent
            if not get_agent(agent_name):
                return (
                    f"Error: Agent '{agent_name}' not found. Create it "
                    f"first with create_agent_config (use list_existing_agents "
                    f"to see what's already available)."
                )

        spec = TriggerSpec(
            id=trigger_id,
            type=type,  # type: ignore[arg-type]
            prompt=prompt,
            agent_name=agent_name or None,
            poll_seconds=poll_seconds,
            path=path or None,
            watch=watch,  # type: ignore[arg-type]
            glob=glob or None,
            script=script or None,
            language=language,  # type: ignore[arg-type]
            match=match or None,
            url=url or None,
            http_mode=http_mode,  # type: ignore[arg-type]
            method=method,  # type: ignore[arg-type]
            json_path=json_path or None,
            repo_path=repo_path or None,
            branch=branch or "HEAD",
            author_filter=author_filter or None,
            path_filter=path_filter or None,
            command=command or None,
            shell_mode=shell_mode,  # type: ignore[arg-type]
            cwd=cwd or None,
        )
        spec_err = validate_spec(spec)
        if spec_err:
            return f"Error: {spec_err}"

        save_trigger(spec)
        register_job(spec.id, spec.poll_seconds)

        if spec.type == "fileos":
            cond = f"file at {spec.path}"
        elif spec.type == "macostool":
            cond = f"osascript ({spec.language}) snippet"
        elif spec.type == "http":
            cond = f"{spec.method} {spec.url} ({spec.http_mode})"
        elif spec.type == "git":
            cond = f"git repo {spec.repo_path} branch {spec.branch}"
        elif spec.type == "shell":
            cond = f"shell command ({spec.shell_mode})"
        else:
            cond = "(unknown)"

        return (
            f"Trigger '{trigger_id}' created. Watching {cond} every "
            f"{spec.poll_seconds}s; will fire "
            f"{agent_name or 'general-purpose agent'}. "
            f"Visible in the Triggers page."
        )

    @tool
    def update_trigger(
        trigger_id: str,
        prompt: str = "",
        poll_seconds: int = 0,
        enabled: bool | None = None,
        agent_name: str = "",
    ) -> str:
        """Update common fields on an existing trigger.

        For trigger-type-specific fields (path, script, regex, …) ask the
        user to delete and recreate — this tool is intentionally narrow
        so users don't accidentally flip a fileos trigger into a
        macostool one.

        Args:
            trigger_id: ID of the trigger to update.
            prompt: New prompt (empty = keep current).
            poll_seconds: New poll interval, 5..86400 (0 = keep current).
            enabled: True/False to flip, or omit to keep.
            agent_name: New worker agent name (empty = keep current).
        """
        from backend.trigger_manager import (
            load_trigger,
            register_job,
            remove_job,
            save_trigger,
            validate_poll_seconds,
        )

        spec = load_trigger(trigger_id)
        if not spec:
            return (
                f"Error: Trigger '{trigger_id}' not found. "
                f"Use list_triggers to see available triggers."
            )

        if prompt:
            spec.prompt = prompt
        if poll_seconds:
            err = validate_poll_seconds(poll_seconds)
            if err:
                return f"Error: {err}"
            spec.poll_seconds = poll_seconds
        if enabled is not None:
            spec.enabled = enabled
        if agent_name:
            from backend.agent_library import get_agent
            if not get_agent(agent_name):
                return f"Error: Agent '{agent_name}' not found."
            spec.agent_name = agent_name

        save_trigger(spec)
        if spec.enabled:
            register_job(spec.id, spec.poll_seconds)
        else:
            remove_job(spec.id)

        status = "enabled" if spec.enabled else "paused"
        return (
            f"Trigger '{trigger_id}' updated ({status}). "
            f"Poll: {spec.poll_seconds}s, agent: "
            f"{spec.agent_name or 'general-purpose'}."
        )

    @tool
    def delete_trigger(trigger_id: str) -> str:
        """Permanently delete a trigger and all its run history.

        Args:
            trigger_id: ID of the trigger to delete.
        """
        from backend.trigger_manager import (
            delete_trigger_files,
            load_trigger,
            remove_job,
        )

        if not load_trigger(trigger_id):
            return f"Error: Trigger '{trigger_id}' not found."

        remove_job(trigger_id)
        delete_trigger_files(trigger_id)
        return f"Trigger '{trigger_id}' deleted."

    @tool
    def run_trigger_now(trigger_id: str) -> str:
        """Fire a trigger immediately, regardless of whether its watched
        condition is currently true.

        Useful for smoke-testing a freshly created trigger — the agent
        runs once with a synthetic event payload so you can verify the
        end-to-end path before waiting for the real condition.

        Args:
            trigger_id: ID of the trigger to run.
        """
        from backend.trigger_manager import (
            load_trigger,
            run_trigger_immediately,
        )

        if not load_trigger(trigger_id):
            return f"Error: Trigger '{trigger_id}' not found."

        run_trigger_immediately(trigger_id)
        return (
            f"Trigger '{trigger_id}' fired manually — the worker agent "
            f"will start momentarily; watch the History page for the "
            f"new session."
        )

    @tool
    def stop_trigger(trigger_id: str) -> str:
        """Cancel a currently running trigger run.  Has no effect if
        nothing is running.

        Args:
            trigger_id: ID of the trigger to stop.
        """
        from backend.trigger_manager import load_trigger, stop_trigger_run

        if not load_trigger(trigger_id):
            return f"Error: Trigger '{trigger_id}' not found."
        cancelled = stop_trigger_run(trigger_id)
        if not cancelled:
            return f"Trigger '{trigger_id}' has no active run to stop."
        return f"Trigger '{trigger_id}' run is being cancelled."

    return [
        list_triggers,
        get_trigger_runs,
        read_trigger_run_output,
        create_trigger,
        update_trigger,
        delete_trigger,
        run_trigger_now,
        stop_trigger,
    ]
