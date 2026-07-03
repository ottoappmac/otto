"""LangChain tools that let the chat agent manage scheduled runs."""

from __future__ import annotations

from langchain_core.tools import tool

from backend.run_output import (
    MAX_RUNS_RETURNED as _MAX_RUNS_RETURNED,
    fmt_duration as _fmt_duration,
    list_run_files as _list_run_files,
    read_run_output as _read_run_output,
)
from backend.scheduler import MAX_SCHEDULES


def build_schedule_tools() -> list:
    """Build schedule management tools for injection into the agent graph."""

    @tool
    def list_schedules() -> str:
        """List all configured schedules with their ID, agent, cron expression,
        enabled status, and last run info."""
        from backend.scheduler import load_all_schedules

        schedules = load_all_schedules()
        if not schedules:
            return "No schedules configured."
        lines = []
        for s in schedules:
            status = "enabled" if s.enabled else "paused"
            last = s.last_run.isoformat() if s.last_run else "never"
            lines.append(
                f"- **{s.id}** ({status}): agent={s.agent_name or 'general-purpose'}, "
                f"cron=`{s.cron_expression}`, last_run={last}, last_status={s.last_status or 'n/a'}"
            )
        lines.append(f"\n{len(schedules)}/{MAX_SCHEDULES} schedule slots used.")
        return "\n".join(lines)

    @tool
    def get_schedule_runs(schedule_id: str, limit: int = 10) -> str:
        """List the run history for a schedule, including each run's outcome.

        Returns, for each past run: run ID, status (success/error/running/
        cancelled), start time, duration, message count, any error, the
        underlying session ID, and a listing of output files the run produced.

        Use this to review how a schedule has been performing over time. To
        read a run's actual output, follow up with:
          - get_session_messages(session_id) for the agent's conversation/result
          - read_schedule_run_output(schedule_id, run_id, file_path) for a file

        Args:
            schedule_id: ID of the schedule (see list_schedules).
            limit: Max number of most-recent runs to return (1–50, default 10).
        """
        from backend.scheduler import load_runs, runs_dir, load_schedule

        if not load_schedule(schedule_id):
            return f"Error: Schedule '{schedule_id}' not found. Use list_schedules to see available schedules."

        limit = max(1, min(limit, _MAX_RUNS_RETURNED))
        runs = load_runs(schedule_id, limit=limit)
        if not runs:
            return f"Schedule '{schedule_id}' has no recorded runs yet."

        lines: list[str] = [f"{len(runs)} run(s) for schedule '{schedule_id}' (most recent first):\n"]
        for r in runs:
            started = r.started_at.strftime("%Y-%m-%d %H:%M") if r.started_at else "unknown"
            duration = _fmt_duration(r.started_at, r.finished_at)
            duration_str = f", {duration}" if duration else ""

            files_dir = runs_dir(schedule_id) / r.id / "files"
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
    def read_schedule_run_output(schedule_id: str, run_id: str, file_path: str = "") -> str:
        """Read an output file produced by a specific scheduled run.

        Use get_schedule_runs first to find the run_id and the names of the
        output files. Leave file_path empty to list the available files for
        the run.

        Args:
            schedule_id: ID of the schedule.
            run_id: ID of the run (from get_schedule_runs).
            file_path: Relative path of the output file to read. Leave empty
                       to list the files available for this run.
        """
        from backend.scheduler import load_schedule, runs_dir

        from backend.utils import is_safe_path_segment

        if not load_schedule(schedule_id):
            return f"Error: Schedule '{schedule_id}' not found. Use list_schedules to see available schedules."
        if not is_safe_path_segment(run_id):
            return f"Error: Run '{run_id}' not found. Use get_schedule_runs to see available runs."

        files_dir = runs_dir(schedule_id) / run_id / "files"
        return _read_run_output(
            files_dir, run_id, file_path, owner_label=f"schedule '{schedule_id}'"
        )

    @tool
    def create_schedule(
        schedule_id: str,
        prompt: str,
        cron_expression: str,
        agent_name: str = "",
    ) -> str:
        """Create a new scheduled agent run.

        Args:
            schedule_id: Unique kebab-case identifier (e.g. "daily-regression")
            prompt: The prompt/instruction the agent will execute each run
            cron_expression: Standard 5-field cron expression (e.g. "0 9 * * 1-5" for weekdays at 9am)
            agent_name: Name of the agent to run (leave empty for general-purpose)
        """
        from apscheduler.triggers.cron import CronTrigger

        from backend.scheduler import (
            get_scheduler,
            load_all_schedules,
            load_schedule,
            register_job,
            save_schedule,
            validate_schedule_id,
        )
        from backend.schemas import ScheduleSpec

        id_err = validate_schedule_id(schedule_id)
        if id_err:
            return f"Error: {id_err}"

        existing = load_all_schedules()
        if len(existing) >= MAX_SCHEDULES:
            return f"Error: Maximum of {MAX_SCHEDULES} schedules reached. Delete one first."

        if load_schedule(schedule_id):
            return f"Error: Schedule '{schedule_id}' already exists."

        try:
            CronTrigger.from_crontab(cron_expression)
        except (ValueError, KeyError) as exc:
            return f"Error: Invalid cron expression: {exc}"

        if agent_name:
            from backend.agent_library import get_agent
            if not get_agent(agent_name):
                return f"Error: Agent '{agent_name}' not found."

        spec = ScheduleSpec(
            id=schedule_id,
            agent_name=agent_name or None,
            prompt=prompt,
            cron_expression=cron_expression,
        )
        save_schedule(spec)

        scheduler = get_scheduler()
        register_job(scheduler, spec.id, spec.cron_expression)

        return (
            f"Schedule '{schedule_id}' created. "
            f"Agent: {agent_name or 'general-purpose'}, Cron: {cron_expression}. "
            f"It will run automatically according to the schedule."
        )

    @tool
    def update_schedule(
        schedule_id: str,
        prompt: str = "",
        cron_expression: str = "",
        agent_name: str = "",
        enabled: bool | None = None,
    ) -> str:
        """Update an existing schedule. Only provided fields are changed.

        Args:
            schedule_id: ID of the schedule to update
            prompt: New prompt (leave empty to keep current)
            cron_expression: New cron expression (leave empty to keep current)
            agent_name: Agent to use for future runs (leave empty to keep current)
            enabled: Set to true/false to enable/disable, or omit to keep current
        """
        from backend.scheduler import (
            get_scheduler,
            load_schedule,
            register_job,
            remove_job,
            save_schedule,
        )

        spec = load_schedule(schedule_id)
        if not spec:
            return f"Error: Schedule '{schedule_id}' not found. Use list_schedules to see available schedules."

        if prompt:
            spec.prompt = prompt
        if cron_expression:
            from apscheduler.triggers.cron import CronTrigger
            try:
                CronTrigger.from_crontab(cron_expression)
            except (ValueError, KeyError) as exc:
                return f"Error: Invalid cron expression: {exc}"
            spec.cron_expression = cron_expression
        if agent_name:
            from backend.agent_library import get_agent
            if not get_agent(agent_name):
                return f"Error: Agent '{agent_name}' not found."
            spec.agent_name = agent_name
        if enabled is not None:
            spec.enabled = enabled

        save_schedule(spec)

        scheduler = get_scheduler()
        if spec.enabled:
            register_job(scheduler, spec.id, spec.cron_expression)
        else:
            remove_job(scheduler, spec.id)

        status = "enabled" if spec.enabled else "paused"
        return f"Schedule '{schedule_id}' updated ({status}). Agent: {spec.agent_name or 'general-purpose'}, Cron: {spec.cron_expression}."

    @tool
    def delete_schedule(schedule_id: str) -> str:
        """Permanently delete a schedule and all its run history.

        Args:
            schedule_id: ID of the schedule to delete
        """
        from backend.scheduler import (
            delete_schedule_files,
            get_scheduler,
            load_schedule,
            remove_job,
        )

        if not load_schedule(schedule_id):
            return f"Error: Schedule '{schedule_id}' not found."

        scheduler = get_scheduler()
        remove_job(scheduler, schedule_id)
        delete_schedule_files(schedule_id)

        return f"Schedule '{schedule_id}' deleted."

    @tool
    def run_schedule_now(schedule_id: str) -> str:
        """Trigger an immediate run of a schedule (does not affect the regular cron timing).

        Args:
            schedule_id: ID of the schedule to run
        """
        from backend.scheduler import load_schedule, run_schedule_immediately

        if not load_schedule(schedule_id):
            return f"Error: Schedule '{schedule_id}' not found."

        run_schedule_immediately(schedule_id)

        return f"Schedule '{schedule_id}' triggered. The run will start momentarily."

    @tool
    def stop_schedule(schedule_id: str) -> str:
        """Stop a currently running scheduled job. Has no effect if nothing is running.

        Args:
            schedule_id: ID of the schedule to stop
        """
        from backend.scheduler import load_schedule, stop_schedule_run

        if not load_schedule(schedule_id):
            return f"Error: Schedule '{schedule_id}' not found."

        cancelled = stop_schedule_run(schedule_id)
        if not cancelled:
            return f"Schedule '{schedule_id}' has no active run to stop."

        return f"Schedule '{schedule_id}' run is being cancelled."

    return [
        list_schedules,
        get_schedule_runs,
        read_schedule_run_output,
        create_schedule,
        update_schedule,
        delete_schedule,
        run_schedule_now,
        stop_schedule,
    ]
