from __future__ import annotations

"""
nepse_date_utils.py
-------------------
Single place for AD ↔ BS conversions.

Preference order:
1) `nepali_datetime` library (if installed) for well-tested conversions
2) Project's built-in conversion tables in `nepal_calendar.py`
"""

from datetime import date, datetime
from typing import Any, Tuple


def ad_to_bs(ad_date: date | datetime) -> Any:
    """Convert AD to BS date (library object if available; else (y, m, d))."""
    d = ad_date.date() if isinstance(ad_date, datetime) else ad_date
    try:
        import nepali_datetime  # type: ignore

        return nepali_datetime.date.from_datetime_date(d)
    except Exception:
        from nepal_calendar import ad_to_bs as _ad_to_bs

        y, m, dd = _ad_to_bs(d)
        return (y, m, dd)


def bs_to_ad(bs_date: Any) -> date:
    """Convert BS to AD date."""
    # If it's a nepali_datetime.date, it has `to_datetime_date()`.
    try:
        fn = getattr(bs_date, "to_datetime_date", None)
        if callable(fn):
            return fn()
    except Exception:
        pass

    # Otherwise accept tuples/lists like (year, month, day).
    if isinstance(bs_date, (tuple, list)) and len(bs_date) >= 3:
        from nepal_calendar import bs_to_ad as _bs_to_ad

        y, m, d = int(bs_date[0]), int(bs_date[1]), int(bs_date[2])
        return _bs_to_ad(y, m, d)

    raise TypeError("Unsupported bs_date type. Provide nepali_datetime.date or (y, m, d).")


def ad_to_bs_ymd(ad_date: date | datetime) -> Tuple[int, int, int]:
    """Always return (y, m, d) tuple for BS date."""
    bs = ad_to_bs(ad_date)
    if isinstance(bs, (tuple, list)) and len(bs) >= 3:
        return (int(bs[0]), int(bs[1]), int(bs[2]))
    return (int(getattr(bs, "year")), int(getattr(bs, "month")), int(getattr(bs, "day")))


def validate_ad_bs_mapping(ad_date: date | datetime) -> None:
    """
    Validation helper: ensure AD date converts to a plausible BS date and back.
    Raises AssertionError on mismatch.
    """
    ad = ad_date.date() if isinstance(ad_date, datetime) else ad_date
    y, m, d = ad_to_bs_ymd(ad)
    assert 1900 <= ad.year <= 2200, f"AD year out of expected range: {ad}"
    assert 2000 <= y <= 2200, f"BS year out of expected range: {(y, m, d)}"
    assert 1 <= m <= 12 and 1 <= d <= 32, f"Invalid BS ymd: {(y, m, d)}"
    # Roundtrip check
    ad2 = bs_to_ad((y, m, d))
    assert ad2 == ad, f"AD↔BS mismatch: AD {ad} -> BS {(y,m,d)} -> AD {ad2}"

