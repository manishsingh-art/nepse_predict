from __future__ import annotations

"""
nepse_calendar.py
-----------------
Thin wrapper around the project's existing `nepal_calendar.py` engine, with an
optional CSV overlay for additional/special NEPSE closures (e.g., system
maintenance, unexpected strikes you want to tag manually).

CSV format (recommended):
  nepse_holidays.csv
    - ad_date (YYYY-MM-DD)
    - reason  (optional free text)
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    from nepal_calendar import NepalMarketCalendar

    _BASE_CAL = NepalMarketCalendar(fetch_live=True)
    _HAS_BASE = True
except Exception:
    _BASE_CAL = None
    _HAS_BASE = False


DEFAULT_HOLIDAYS_CSV = "nepse_holidays.csv"


def _normalize_d(d: date | datetime) -> date:
    return d.date() if hasattr(d, "date") and not isinstance(d, date) else d  # type: ignore[arg-type]


def _load_csv_holidays(path: str | Path = DEFAULT_HOLIDAYS_CSV) -> set[date]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        df = pd.read_csv(p, parse_dates=["ad_date"])
        if "ad_date" not in df.columns:
            return set()
        return {d.date() for d in pd.to_datetime(df["ad_date"]).dropna().dt.to_pydatetime()}
    except Exception:
        return set()


@dataclass(frozen=True)
class TradingCalendarOverlay:
    """
    Overlay calendar:
    - Base truth from `NepalMarketCalendar` (weekends Fri/Sat, public holidays, bandh detection).
    - Extra closures from a CSV (user-maintained).
    """

    holidays_csv: str | Path = DEFAULT_HOLIDAYS_CSV

    def is_trading_day(self, d: date | datetime) -> bool:
        dd = _normalize_d(d)
        if dd in _load_csv_holidays(self.holidays_csv):
            return False
        if _HAS_BASE and _BASE_CAL is not None:
            return bool(_BASE_CAL.is_trading_day(dd))
        # Fallback rule (best-effort): Sun–Thu open, Fri/Sat closed
        return dd.weekday() not in (4, 5)

    def next_n_trading_dates(self, from_date: date | datetime, n: int) -> list[date]:
        start = _normalize_d(from_date)
        out: list[date] = []
        d = start
        while len(out) < n:
            d = d + timedelta(days=1)
            if self.is_trading_day(d):
                out.append(d)
            if (d - start).days > 90:
                break
        return out


def next_trading_days(
    start_date: datetime,
    n_days: int,
    holidays_csv: str | Path = DEFAULT_HOLIDAYS_CSV,
) -> list[datetime]:
    """
    Generate next N valid NEPSE trading days (as datetimes at 00:00).

    Uses `NepalMarketCalendar` when available (preferred). Also supports a
    user-maintained CSV overlay for additional closures.
    """
    cal = TradingCalendarOverlay(holidays_csv=holidays_csv)
    days = cal.next_n_trading_dates(start_date, n_days)
    return [datetime.combine(d, datetime.min.time()) for d in days]


def iter_trading_days(
    start_date: date | datetime,
    end_date: date | datetime,
    holidays_csv: str | Path = DEFAULT_HOLIDAYS_CSV,
) -> Iterable[date]:
    """Yield trading days in [start_date, end_date] inclusive."""
    cal = TradingCalendarOverlay(holidays_csv=holidays_csv)
    s = _normalize_d(start_date)
    e = _normalize_d(end_date)
    d = s
    while d <= e:
        if cal.is_trading_day(d):
            yield d
        d += timedelta(days=1)

