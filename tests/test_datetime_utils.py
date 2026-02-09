from datetime import datetime, timezone

from src.utils.datetime_utils import parse_iso_datetime_to_utc


def test_parse_iso_datetime_to_utc_handles_offset_z_and_naive():
    base = datetime(2026, 2, 9, 12, 34, 56, tzinfo=timezone.utc)

    # Explicit UTC offset
    assert parse_iso_datetime_to_utc("2026-02-09T12:34:56+00:00") == base

    # Trailing Z
    assert parse_iso_datetime_to_utc("2026-02-09T12:34:56Z") == base

    # Naive (assume UTC)
    parsed_naive = parse_iso_datetime_to_utc("2026-02-09T12:34:56")
    assert parsed_naive == base
    assert parsed_naive.tzinfo is timezone.utc

    # Non-UTC offset converts correctly
    assert parse_iso_datetime_to_utc("2026-02-09T14:34:56+02:00") == base


def test_parse_iso_datetime_to_utc_rejects_invalid():
    assert parse_iso_datetime_to_utc("") is None
    assert parse_iso_datetime_to_utc("   ") is None
    assert parse_iso_datetime_to_utc(None) is None
    assert parse_iso_datetime_to_utc("not-a-date") is None
