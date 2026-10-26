"""Date handling for chat requests ("today", "last 7 days", "Oct 1 to Oct 10").

The LLM only says WHICH kind of period the user asked for; the actual dates are
computed and validated here, deterministically, in the business timezone. Anything
that cannot be turned into a clear, valid range becomes a clarification question
instead of a guess."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DateRange:
    start: date  # inclusive, business-timezone calendar day
    end: date  # inclusive, business-timezone calendar day

    def label(self) -> str:
        if self.start == self.end:
            return self.start.isoformat()
        return f"{self.start.isoformat()} to {self.end.isoformat()}"

    def utc_bounds(self, tz: ZoneInfo) -> tuple[datetime, datetime]:
        """[start of first day, start of the day after the last day) as aware datetimes.
        The API layer converts them to UTC."""
        start = datetime.combine(self.start, time.min, tzinfo=tz)
        end = datetime.combine(self.end + timedelta(days=1), time.min, tzinfo=tz)
        return start, end


class DateRangeError(ValueError):
    """The requested period is unclear or invalid. The message is a question/explanation
    that can be shown to the user as-is."""


def today_in(tz: ZoneInfo) -> date:
    return datetime.now(tz).date()


def parse_iso_date(value: str | None, what: str) -> date:
    if not value:
        raise DateRangeError(f"Which {what} date do you mean? Please give it as a full date, e.g. 2026-10-01.")
    try:
        return date.fromisoformat(value.strip())
    except ValueError:
        raise DateRangeError(f"I couldn't read the {what} date '{value}'. Please give it as YYYY-MM-DD.") from None


def resolve_period(
    period: str,
    today: date,
    max_range_days: int,
    last_n_days: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> DateRange | None:
    """Turn the interpreted period into a validated DateRange.


    period:
      - "none" / "all" -> None (no date filter). The graph treats "none" (no dates in the
                          message) as "keep the conversation's dates" before calling this.
      - "today"       -> today
      - "yesterday"   -> yesterday
      - "last_n_days" -> the last N calendar days INCLUDING today (last 7 days = today and the 6 before)
      - "date_range"  -> start_date..end_date (inclusive); a single day when end_date is missing
    """
    if period in ("none", "all"):
        return None
    if period == "today":
        return DateRange(today, today)
    if period == "yesterday":
        day = today - timedelta(days=1)
        return DateRange(day, day)
    if period == "last_n_days":
        if not last_n_days or last_n_days < 1:
            raise DateRangeError("How many days back should I look?")
        if last_n_days > max_range_days:
            raise DateRangeError(f"Please ask for {max_range_days} days or fewer.")
        return DateRange(today - timedelta(days=last_n_days - 1), today)
    if period == "date_range":
        start = parse_iso_date(start_date, "start")
        end = parse_iso_date(end_date, "end") if end_date else start
        return validate_range(DateRange(start, end), today, max_range_days)
    raise DateRangeError("Which dates should I look at? For example: today, the last 7 days, or October 1 to October 10.")


def validate_range(requested: DateRange, today: date, max_range_days: int) -> DateRange:
    if requested.start > requested.end:
        raise DateRangeError(
            f"The start date {requested.start.isoformat()} is after the end date {requested.end.isoformat()}. "
            "Which range did you mean?"
        )
    if requested.start > today:
        raise DateRangeError(
            f"{requested.start.isoformat()} is in the future, so there can't be enquiries yet. Which dates did you mean?"
        )
    if (requested.end - requested.start).days + 1 > max_range_days:
        raise DateRangeError(f"That range is longer than {max_range_days} days. Please choose a shorter range.")
    # An end date in the future is harmless (no enquiries exist yet); cap it at today so
    # the summary reports the range that was actually searched.
    return DateRange(requested.start, min(requested.end, today))
