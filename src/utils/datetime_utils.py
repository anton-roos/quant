from __future__ import annotations

from datetime import datetime, timezone


def parse_iso_datetime_to_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 datetime string and normalize it to UTC (tz-aware).

    Accepts timestamps with:
    - Explicit offsets, e.g. "2026-02-09T12:34:56+00:00"
    - Trailing 'Z', e.g. "2026-02-09T12:34:56Z"
    - Naive ISO strings (assumed UTC), e.g. "2026-02-09T12:34:56"

    Returns None if parsing fails or value is falsy.
    """
    if not value:
        return None

    s = value.strip()
    if not s:
        return None

    # Python's datetime.fromisoformat doesn't accept a trailing 'Z'.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
