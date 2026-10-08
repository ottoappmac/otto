"""Fresh calendar context and deterministic weekday checks.

Models invent weekdays (a Friday that is actually a Saturday). The
middleware injects :func:`build_date_block` on every model call, and
:func:`find_date_mismatches` checks weekday/date pairs in answers and
file writes against the real calendar.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# Weekday token -> Monday=0 .. Sunday=6 (datetime.date.weekday()).
_WEEKDAY_INDEX: dict[str, int] = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_MONTH_INDEX: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

_WD = (
    r"monday|mon|tuesday|tue|tues|wednesday|wed|thursday|thu|thur|thurs|"
    r"friday|fri|saturday|sat|sunday|sun"
)
_MON = (
    r"january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|"
    r"august|aug|september|sept|sep|october|oct|november|nov|december|dec"
)
_ORD = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:\s*,?\s*(\d{4}))?"

# "Friday 7 November", "Sat, 8 Nov 2026", "Friday (7 Nov)".
_PAIR_DM = re.compile(
    rf"\b({_WD})\b\.?\s*(?:,\s*|\(\s*)?(\d{{1,2}}){_ORD}\s+({_MON})\b{_YEAR}",
    re.IGNORECASE,
)
# "Friday, November 7, 2026".
_PAIR_MD = re.compile(
    rf"\b({_WD})\b\.?,?\s+({_MON})\b\s+(\d{{1,2}}){_ORD}{_YEAR}",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")

_DATE_BLOCK_START = "<current_date>"
_DATE_BLOCK_END = "</current_date>"

# How many months of calendar to spell out, including the current month.
_CALENDAR_MONTHS = 14


@dataclass(frozen=True)
class DateMismatch:
    """A weekday that does not fall on the stated calendar date."""

    span: str
    claimed: str
    actual: str
    iso: str

    def line(self) -> str:
        return f"{self.iso} is a {self.actual}, not {self.claimed} ({self.span!r})."


def build_date_block(now: datetime | None = None) -> str:
    """Today's date plus a compact calendar covering the next 14 months.

    The block tells the model not to do weekday arithmetic itself.
    """
    now = now or datetime.now().astimezone()
    tz = now.tzname() or "local"
    lines = [
        _DATE_BLOCK_START,
        f"Today is {now.strftime('%A, %B %d, %Y')} {now.strftime('%H:%M')} {tz}.",
        "Never compute weekdays mentally. Look them up in this calendar or call `check_dates`.",
        "When you name a weekday with a date, the weekday must be the one shown here.",
        "",
    ]
    year, month = now.year, now.month
    for _ in range(_CALENDAR_MONTHS):
        lines.append(_month_calendar(year, month))
        lines.append("")
        month += 1
        if month == 13:
            month = 1
            year += 1
    lines.append(_DATE_BLOCK_END)
    return "\n".join(lines).strip()


def strip_date_block(text: str) -> str:
    """Remove a previously injected calendar block, if present."""
    start = text.find(_DATE_BLOCK_START)
    end = text.find(_DATE_BLOCK_END)
    if start == -1 or end == -1 or end < start:
        return text
    end += len(_DATE_BLOCK_END)
    return (text[:start] + text[end:]).strip()


def _month_calendar(year: int, month: int) -> str:
    cal = calendar.Calendar(firstweekday=calendar.SUNDAY)
    weeks = cal.monthdayscalendar(year, month)
    title = f"{calendar.month_name[month]} {year}"
    rows = ["Su Mo Tu We Th Fr Sa"]
    for week in weeks:
        rows.append(" ".join(f"{day:2d}" if day else "  " for day in week))
    return title + "\n" + "\n".join(rows)


def find_date_mismatches(text: str, now: datetime | None = None) -> list[DateMismatch]:
    """Return weekday/date pairs in *text* whose weekday is wrong.

    A missing year uses the nearest explicit year in *text*, otherwise the
    next upcoming occurrence of that month and day. Phrases with no day
    and month (``Friday night``) are ignored.
    """
    if not text:
        return []
    now = now or datetime.now().astimezone()
    found: list[DateMismatch] = []
    occupied: list[tuple[int, int]] = []
    for pattern, month_first in ((_PAIR_DM, False), (_PAIR_MD, True)):
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if any(span[0] < end and span[1] > start for start, end in occupied):
                continue
            mismatch = _mismatch_from_match(text, match, month_first=month_first, now=now)
            occupied.append(span)
            if mismatch is not None:
                found.append(mismatch)
    return found


def format_date_correction(mismatches: list[DateMismatch]) -> str:
    """The one-shot note handed back to the model."""
    lines = ["Date check failed. These weekday and date pairs do not match:"]
    for item in mismatches:
        lines.append(f"- {item.line()}")
    lines.append(
        "Correct every weekday so it matches the real calendar. "
        "Do not change the calendar dates themselves."
    )
    return "\n".join(lines)


def describe_dates(query: str, now: datetime | None = None) -> str:
    """Resolve one or more date phrases for the ``check_dates`` tool."""
    now = now or datetime.now().astimezone()
    today = now.date()
    parts = [part.strip() for part in re.split(r"[\n;]+", query) if part.strip()]
    if not parts:
        return "Error: pass one or more dates, separated by semicolons or newlines."
    lines: list[str] = []
    for part in parts:
        resolved = _parse_one(part, today)
        if isinstance(resolved, str):
            lines.append(f"{part}: {resolved}")
        else:
            lines.append(_format_resolved(part, resolved, today))
    return "\n".join(lines)


def _mismatch_from_match(
    text: str,
    match: re.Match[str],
    *,
    month_first: bool,
    now: datetime,
) -> DateMismatch | None:
    claimed_token = match.group(1)
    if month_first:
        month_token, day_token, year_token = match.group(2), match.group(3), match.group(4)
    else:
        day_token, month_token, year_token = match.group(2), match.group(3), match.group(4)
    claimed_idx = _WEEKDAY_INDEX.get(claimed_token.lower())
    month = _MONTH_INDEX.get(month_token.lower())
    if claimed_idx is None or month is None:
        return None
    day = int(day_token)
    if year_token:
        year = int(year_token)
        try:
            resolved = date(year, month, day)
        except ValueError:
            return None
    else:
        year = _nearest_year(text, match.start())
        if year is not None:
            try:
                resolved = date(year, month, day)
            except ValueError:
                return None
        else:
            resolved = _upcoming(month, day, now.date())
            if resolved is None:
                return None
    actual_idx = resolved.weekday()
    if actual_idx == claimed_idx:
        return None
    return DateMismatch(
        span=match.group(0).strip(),
        claimed=calendar.day_name[claimed_idx],
        actual=calendar.day_name[actual_idx],
        iso=resolved.isoformat(),
    )


def _nearest_year(text: str, anchor: int) -> int | None:
    best: int | None = None
    best_dist: int | None = None
    for match in _YEAR_RE.finditer(text):
        dist = abs(match.start() - anchor)
        if best_dist is None or dist < best_dist:
            best = int(match.group(1))
            best_dist = dist
    return best


def _upcoming(month: int, day: int, today: date) -> date | None:
    for year in (today.year, today.year + 1):
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


def _parse_one(text: str, today: date) -> date | str:
    low = " ".join(text.strip().lower().replace(",", " ").split())
    if low in ("today", "now"):
        return today
    if low == "tomorrow":
        return today + timedelta(days=1)
    if low == "yesterday":
        return today - timedelta(days=1)

    relative = re.fullmatch(r"in\s+(\d+)\s+days?", low)
    if relative:
        return today + timedelta(days=int(relative.group(1)))
    ago = re.fullmatch(r"(\d+)\s+days?\s+ago", low)
    if ago:
        return today - timedelta(days=int(ago.group(1)))

    named = re.fullmatch(r"(next|this)\s+([a-z]+)", low)
    if named:
        idx = _WEEKDAY_INDEX.get(named.group(2))
        if idx is None:
            return "could not parse"
        return _named_weekday(idx, today, strictly_after=named.group(1) == "next")

    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", low)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return "not a real date"

    parsed = _parse_calendar_date(low, today)
    if parsed is None:
        return "could not parse"
    return parsed


def _named_weekday(index: int, today: date, *, strictly_after: bool) -> date:
    delta = (index - today.weekday()) % 7
    if delta == 0 and strictly_after:
        delta = 7
    return today + timedelta(days=delta)


def _parse_calendar_date(low: str, today: date) -> date | None:
    """``7 november 2026``, ``november 7 2026``, ``7 nov``, ``nov 7``."""
    day_month = re.fullmatch(
        rf"(\d{{1,2}}){_ORD}\s+({_MON})(?:\s+(\d{{4}}))?",
        low,
    )
    month_day = re.fullmatch(
        rf"({_MON})\s+(\d{{1,2}}){_ORD}(?:\s+(\d{{4}}))?",
        low,
    )
    if day_month:
        day_s, month_s, year_s = day_month.group(1), day_month.group(2), day_month.group(3)
    elif month_day:
        month_s, day_s, year_s = month_day.group(1), month_day.group(2), month_day.group(3)
    else:
        return None
    month = _MONTH_INDEX.get(month_s)
    if month is None:
        return None
    day = int(day_s)
    if year_s:
        try:
            return date(int(year_s), month, day)
        except ValueError:
            return None
    return _upcoming(month, day, today)


def _format_resolved(source: str, resolved: date, today: date) -> str:
    delta = (resolved - today).days
    if delta == 0:
        rel = "today"
    elif delta > 0:
        rel = f"in {delta} day" if delta == 1 else f"in {delta} days"
    else:
        n = abs(delta)
        rel = f"{n} day ago" if n == 1 else f"{n} days ago"
    weekday = calendar.day_name[resolved.weekday()]
    return f"{source}: {resolved.isoformat()} {weekday} ({rel})"
