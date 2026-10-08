"""Deterministic date and weekday lookup for the agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from utilities.logger import get_logger

logger = get_logger()


class CheckDatesTool(BaseTool):
    """Resolve dates and relative phrases to a weekday and ISO date."""

    name: str = "check_dates"
    description: str = (
        "Look up the real weekday for one or more dates. Pass dates separated "
        "by semicolons or newlines, for example '2026-11-07', '7 November 2026', "
        "'next Friday', or 'in 3 days'. Returns the weekday, ISO date, and days "
        "from today. Use this instead of guessing which weekday a date falls on."
    )

    def _run(self, dates: str) -> str:
        from backend.date_context import describe_dates

        logger.info("check_dates: %s", dates)
        return describe_dates(dates)

    async def _arun(self, dates: str) -> str:
        return self._run(dates)
