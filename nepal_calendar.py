#!/usr/bin/env python3
"""
nepal_calendar.py — Nepal Market Calendar & Holiday Engine
===========================================================
Provides:
  - Bikram Sambat (BS) ↔ Gregorian (AD) conversion
  - Government public holidays (NRB-published schedule)
  - Nepal Bandh (general strike) detection via news RSS
  - NEPSE trading session rules (Sun–Thu, 11:00–15:00 NST)
  - Feature vector for ML models (28 calendar features)

NEPSE trading days: Sunday–Thursday (Friday–Saturday closed)
Market hours: 11:00–15:00 Nepal Standard Time (UTC+5:45)

Usage:
    from nepal_calendar import NepalMarketCalendar, ad_to_bs, bs_to_ad
    cal = NepalMarketCalendar()
    cal.is_trading_day(date.today())
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Bikram Sambat Conversion Tables ─────────────────────────────────────────
# Days in each BS month for years 2000–2090 BS
# Source: Nepal Government calendar data
_BS_MONTH_DATA: Dict[int, List[int]] = {
    2000: [30,32,31,32,31,30,30,30,29,30,29,31],
    2001: [31,31,32,31,31,31,30,29,30,29,30,30],
    2002: [31,31,32,32,31,30,30,29,30,29,30,30],
    2003: [31,32,31,32,31,30,30,30,29,29,30,31],
    2004: [30,32,31,32,31,30,30,30,29,30,29,31],
    2005: [31,31,32,31,31,31,30,29,30,29,30,30],
    2006: [31,31,32,32,31,30,30,29,30,29,30,30],
    2007: [31,32,31,32,31,30,30,30,29,29,30,31],
    2008: [31,31,32,31,31,31,30,29,30,29,30,30],
    2009: [31,31,32,32,31,30,30,29,30,29,30,30],
    2010: [31,32,31,32,31,30,30,30,29,29,30,31],
    2011: [31,31,32,31,31,31,30,29,30,29,30,30],
    2012: [31,31,32,32,31,30,30,29,30,29,30,30],
    2013: [31,32,31,32,31,30,30,30,29,30,29,31],
    2014: [31,31,32,31,31,31,30,29,30,29,30,30],
    2015: [31,31,32,32,31,30,30,30,29,29,30,31],
    2016: [31,32,31,32,31,30,30,30,29,30,29,31],
    2017: [31,31,32,31,31,31,30,29,30,29,30,30],
    2018: [31,31,32,32,31,30,30,29,30,29,30,30],
    2019: [31,32,31,32,31,30,30,30,29,30,29,31],
    2020: [31,31,32,31,31,31,30,29,30,29,30,30],
    2021: [31,31,32,32,31,30,30,30,29,29,30,31],
    2022: [31,32,31,32,31,30,30,30,29,30,29,31],
    2023: [31,31,32,31,31,31,30,29,30,29,30,30],
    2024: [31,31,32,32,31,30,30,29,30,29,30,30],
    2025: [31,32,31,32,31,30,30,30,29,30,29,31],
    2026: [31,31,32,31,31,31,30,29,30,29,30,30],
    2027: [31,31,32,32,31,30,30,30,29,29,30,31],
    2028: [31,32,31,32,31,30,30,30,29,30,29,31],
    2029: [31,31,32,31,31,31,30,29,30,29,30,30],
    2030: [31,31,32,32,31,30,30,29,30,29,30,30],
    2031: [31,32,31,32,31,30,30,30,29,30,29,31],
    2032: [31,31,32,31,31,31,30,29,30,29,30,30],
    2033: [31,31,32,32,31,30,30,30,29,29,30,31],
    2034: [31,32,31,32,31,30,30,30,29,30,29,31],
    2035: [31,31,32,31,31,31,30,29,30,29,30,30],
    2036: [31,31,32,32,31,30,30,29,30,29,30,30],
    2037: [31,32,31,32,31,30,30,30,29,30,29,31],
    2038: [31,31,32,31,31,31,30,29,30,29,30,30],
    2039: [31,31,32,32,31,30,30,30,29,29,30,31],
    2040: [31,32,31,32,31,30,30,30,29,30,29,31],
    2041: [31,31,32,31,31,31,30,29,30,29,30,30],
    2042: [31,31,32,32,31,30,30,29,30,29,30,30],
    2043: [31,32,31,32,31,30,30,30,29,30,29,31],
    2044: [31,31,32,31,31,31,30,29,30,29,30,30],
    2045: [31,31,32,32,31,30,30,30,29,29,30,31],
    2046: [31,32,31,32,31,30,30,30,29,30,29,31],
    2047: [31,31,32,31,31,31,30,29,30,29,30,30],
    2048: [31,32,31,32,31,30,30,29,30,29,30,30],
    2049: [31,32,31,32,31,30,30,30,29,30,29,31],
    2050: [31,31,32,31,31,31,30,29,30,29,30,30],
    2051: [31,31,32,32,31,30,30,30,29,29,30,31],
    2052: [31,32,31,32,31,30,30,30,29,30,29,31],
    2053: [31,31,32,31,31,31,30,29,30,29,30,30],
    2054: [31,31,32,32,31,30,30,29,30,29,30,30],
    2055: [31,32,31,32,31,30,30,30,29,30,29,31],
    2056: [31,31,32,31,31,31,30,29,30,29,30,30],
    2057: [31,31,32,32,31,30,30,30,29,29,30,31],
    2058: [31,32,31,32,31,30,30,30,29,30,29,31],
    2059: [31,31,32,31,31,31,30,29,30,29,30,30],
    2060: [31,31,32,32,31,30,30,29,30,29,30,30],
    2061: [31,32,31,32,31,30,30,30,29,30,29,31],
    2062: [31,31,32,31,31,31,30,29,30,29,30,30],
    2063: [31,31,32,32,31,30,30,30,29,29,30,31],
    2064: [31,32,31,32,31,30,30,30,29,30,29,31],
    2065: [31,31,32,31,31,31,30,29,30,29,30,30],
    2066: [31,31,32,32,31,30,30,29,30,29,30,30],
    2067: [31,32,31,32,31,30,30,30,29,30,29,31],
    2068: [31,31,32,31,31,31,30,29,30,29,30,30],
    2069: [31,31,32,32,31,30,30,30,29,29,30,31],
    2070: [31,32,31,32,31,30,30,30,29,30,29,31],
    2071: [31,31,32,31,31,31,30,29,30,29,30,30],
    2072: [31,31,32,32,31,30,30,29,30,29,30,30],
    2073: [31,32,31,32,31,30,30,30,29,30,29,31],
    2074: [31,31,32,31,31,31,30,29,30,29,30,30],
    2075: [31,31,32,32,31,30,30,30,29,29,30,31],
    2076: [31,32,31,32,31,30,30,30,29,30,29,31],
    2077: [31,31,32,31,31,31,30,29,30,29,30,30],
    2078: [31,31,32,32,31,30,30,29,30,29,30,30],
    2079: [31,32,31,32,31,30,30,30,29,30,29,31],
    2080: [31,31,32,31,31,31,30,29,30,29,30,30],
    2081: [31,31,32,32,31,30,30,30,29,29,30,31],
    2082: [31,32,31,32,31,30,30,30,29,30,29,31],
    2083: [31,31,32,31,31,31,30,29,30,29,30,30],
    2084: [31,31,32,32,31,30,30,29,30,29,30,30],
    2085: [31,32,31,32,31,30,30,30,29,30,29,31],
    2086: [31,31,32,31,31,31,30,29,30,29,30,30],
    2087: [31,31,32,32,31,30,30,30,29,29,30,31],
    2088: [31,32,31,32,31,30,30,30,29,30,29,31],
    2089: [31,31,32,31,31,31,30,29,30,29,30,30],
    2090: [31,31,32,32,31,30,30,29,30,29,30,30],
}

# BS epoch: BS 2000/01/01 = AD 1943/04/14
_BS_EPOCH_AD = date(1943, 4, 14)
_BS_EPOCH_BS = (2000, 1, 1)

BS_MONTHS = [
    "Baisakh", "Jestha", "Ashadh", "Shrawan",
    "Bhadra", "Ashwin", "Kartik", "Mangsir",
    "Poush", "Magh", "Falgun", "Chaitra",
]


def get_bs_month_name(month: int) -> str:
    return BS_MONTHS[month - 1] if 1 <= month <= 12 else "Unknown"


def ad_to_bs(ad_date: date) -> Tuple[int, int, int]:
    """Convert Gregorian date to Bikram Sambat. Returns (year, month, day)."""
    delta = (ad_date - _BS_EPOCH_AD).days
    bs_year, bs_month, bs_day = _BS_EPOCH_BS

    while delta > 0:
        year_data = _BS_MONTH_DATA.get(bs_year)
        if year_data is None:
            raise ValueError(f"BS year {bs_year} not in conversion table.")
        days_in_month = year_data[bs_month - 1]
        if delta < days_in_month - bs_day + 1:
            bs_day += delta
            delta = 0
        else:
            delta -= (days_in_month - bs_day + 1)
            bs_day = 1
            bs_month += 1
            if bs_month > 12:
                bs_month = 1
                bs_year += 1
    return bs_year, bs_month, bs_day


def bs_to_ad(bs_year: int, bs_month: int, bs_day: int) -> date:
    """Convert Bikram Sambat date to Gregorian."""
    total_days = 0
    cur_year, cur_month, cur_day = _BS_EPOCH_BS

    # Count days from epoch to target
    while (cur_year, cur_month, cur_day) != (bs_year, bs_month, bs_day):
        year_data = _BS_MONTH_DATA.get(cur_year)
        if year_data is None:
            raise ValueError(f"BS year {cur_year} not in table.")
        days_in_month = year_data[cur_month - 1]
        if cur_day < days_in_month:
            cur_day += 1
        else:
            cur_day = 1
            cur_month += 1
            if cur_month > 12:
                cur_month = 1
                cur_year += 1
        total_days += 1
        if total_days > 50000:
            raise ValueError("BS to AD conversion diverged — check inputs.")

    return _BS_EPOCH_AD + timedelta(days=total_days)


# ─── Static Public Holidays ───────────────────────────────────────────────────
# NRB-published NEPSE trading holidays (AD dates, updated annually)
# Format: (month, day) for recurring annual holidays
# And specific-year dates for variable holidays

_FIXED_ANNUAL_HOLIDAYS: List[Tuple[int, int]] = [
    # These are AD month/day for common Nepal public holidays
    # (many are lunar-based and shift annually — see _variable_holidays for those)
    (1, 11),   # Prithvi Narayan Shah Birthday (approx)
    (9, 20),   # Constitution Day
]

# Variable holidays by BS year → list of AD dates
# Updated through BS 2082 (AD 2025-2026)
_VARIABLE_HOLIDAYS_BY_YEAR: Dict[int, List[str]] = {
    2081: [  # AD 2024-2025
        "2024-04-14",  # Nepali New Year (BS 2081 Baisakh 1)
        "2024-04-19",  # Ram Navami
        "2024-05-23",  # Buddha Jayanti
        "2024-07-17",  # Ashad 15 (National Paddy Day)
        "2024-08-19",  # Janai Purnima
        "2024-08-26",  # Gai Jatra
        "2024-09-16",  # Indra Jatra
        "2024-10-02",  # Dashain (Ghatasthapana)
        "2024-10-12",  # Dashain (Maha Ashtami)
        "2024-10-13",  # Dashain (Maha Navami)
        "2024-10-14",  # Vijaya Dashami
        "2024-10-15",  # Dashain (Ekadashi)
        "2024-10-17",  # Fulpati
        "2024-11-01",  # Tihar (Laxmi Puja)
        "2024-11-02",  # Tihar (Gobardhan Puja)
        "2024-11-03",  # Bhai Tika
        "2024-11-07",  # Chhath Parva
        "2024-12-25",  # Christmas (observed)
        "2025-01-14",  # Maghe Sankranti
        "2025-01-20",  # Sonam Lhosar
        "2025-02-19",  # National Democracy Day
        "2025-03-04",  # Maha Shivaratri
        "2025-03-14",  # Holi (Fagu Purnima)
        "2025-04-01",  # Ghode Jatra
        "2025-04-02",  # Ram Navami
        "2025-04-13",  # Chaitra Dasain
        "2025-04-14",  # Nepali New Year BS 2082
    ],
    2082: [  # AD 2025-2026
        "2025-04-14",  # Nepali New Year (BS 2082 Baisakh 1)
        "2025-05-12",  # Buddha Jayanti
        "2025-07-05",  # Ashad 15
        "2025-08-09",  # Janai Purnima
        "2025-08-15",  # Gai Jatra
        "2025-09-06",  # Indra Jatra
        "2025-09-22",  # Dashain (Ghatasthapana)
        "2025-10-01",  # Dashain (Maha Ashtami)
        "2025-10-02",  # Dashain (Maha Navami)
        "2025-10-03",  # Vijaya Dashami
        "2025-10-04",  # Dashain (Ekadashi)
        "2025-10-20",  # Tihar (Laxmi Puja)
        "2025-10-21",  # Tihar (Gobardhan Puja)
        "2025-10-22",  # Bhai Tika
        "2025-10-27",  # Chhath Parva
        "2026-01-15",  # Maghe Sankranti
        "2026-02-07",  # Sonam Lhosar
        "2026-02-19",  # National Democracy Day
        "2026-02-26",  # Maha Shivaratri
        "2026-03-03",  # Holi
        "2026-03-22",  # Ghode Jatra
        "2026-03-23",  # Ram Navami
        "2026-04-01",  # Chaitra Dasain
        "2026-04-14",  # Nepali New Year BS 2083
    ],
}

# Special exchange closures (NRB directives, system maintenance, etc.)
_SPECIAL_CLOSURES: List[str] = [
    # Add NRB-notified special closures here
    "2025-05-01",  # Labour Day (observed)
    "2025-12-31",  # Year-end (if declared)
]


def _load_all_holidays() -> frozenset:
    """Build a frozen set of all AD date strings for non-trading days."""
    holidays: set = set()

    for year_holidays in _VARIABLE_HOLIDAYS_BY_YEAR.values():
        holidays.update(year_holidays)

    holidays.update(_SPECIAL_CLOSURES)

    return frozenset(holidays)


_HOLIDAY_SET: frozenset = _load_all_holidays()


# ─── Nepal Bandh Detection ────────────────────────────────────────────────────

_BANDH_KEYWORDS = [
    "nepal bandh", "banda", "bandh", "shutdown", "general strike",
    "transport strike", "chakka jam", "market closed", "nepal strike",
    "बन्द", "हड़ताल", "आम हड़ताल",
]

_BANDH_CACHE_FILE = Path.home() / ".nepse_cache" / "bandh_dates.json"


def _load_bandh_cache() -> Dict[str, bool]:
    """Load cached bandh dates from disk."""
    try:
        if _BANDH_CACHE_FILE.exists():
            age = time.time() - _BANDH_CACHE_FILE.stat().st_mtime
            if age < 3600 * 6:  # fresh within 6 hours
                return json.loads(_BANDH_CACHE_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_bandh_cache(data: Dict[str, bool]) -> None:
    try:
        _BANDH_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _BANDH_CACHE_FILE.write_text(json.dumps(data))
    except Exception:
        pass


def detect_bandh_from_news(target_date: date, fetch_live: bool = True) -> bool:
    """
    Check if a Nepal Bandh is active on `target_date` by scanning news RSS feeds.
    Returns True if a bandh is detected for that date.
    Falls back to False if no network.
    """
    cache = _load_bandh_cache()
    date_str = target_date.isoformat()
    if date_str in cache:
        return cache[date_str]

    if not fetch_live:
        return False

    RSS_FEEDS = [
        "https://thehimalayantimes.com/feed/",
        "https://english.onlinekhabar.com/feed",
        "https://myrepublica.nagariknetwork.com/feed/",
        "https://kathmandupost.com/rss",
    ]

    target_str = target_date.strftime("%Y-%m-%d")
    found_bandh = False

    for feed_url in RSS_FEEDS:
        try:
            import requests
            r = requests.get(feed_url, timeout=8,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            items = re.findall(r"<item>(.*?)</item>", r.text, re.DOTALL)
            for item in items:
                title = re.search(r"<title[^>]*>(.*?)</title>", item, re.DOTALL)
                pub = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
                if not title or not pub:
                    continue
                title_text = title.group(1).lower()
                pub_text = pub.group(1)
                # Match keywords
                if any(kw in title_text for kw in _BANDH_KEYWORDS):
                    # Check if pub date is within ±2 days of target
                    for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z"]:
                        try:
                            from datetime import datetime
                            pub_dt = datetime.strptime(pub_text.strip(), fmt).date()
                            if abs((pub_dt - target_date).days) <= 2:
                                found_bandh = True
                                break
                        except Exception:
                            continue
                if found_bandh:
                    break
            if found_bandh:
                break
        except Exception:
            continue

    cache[date_str] = found_bandh
    _save_bandh_cache(cache)
    return found_bandh


# ─── Calendar Engine ──────────────────────────────────────────────────────────

class NepalMarketCalendar:
    """
    NEPSE trading calendar with full Nepal holiday support.

    Trading rules:
      - Trading days: Sunday through Thursday (NEPSE standard)
      - Closed: Friday, Saturday (weekend)
      - Closed: All NRB-declared public holidays
      - Closed: Nepal Bandh days (detected from news)
      - Market hours: 11:00 – 15:00 NST (UTC+5:45)

    Usage:
        cal = NepalMarketCalendar(fetch_live=True)
        cal.is_trading_day(date.today())
        cal.next_n_trading_dates(date.today(), 7)
    """

    # NEPSE weekdays: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
    # Trading: Sun(6), Mon(0), Tue(1), Wed(2), Thu(3)
    TRADING_WEEKDAYS = {6, 0, 1, 2, 3}  # Sunday=6 in Python's weekday()
    CLOSED_WEEKDAYS  = {4, 5}            # Friday=4, Saturday=5

    def __init__(self, fetch_live: bool = True):
        self.fetch_live = fetch_live
        self._holiday_set = _HOLIDAY_SET
        self._bandh_cache: Dict[str, bool] = _load_bandh_cache()

    def is_weekend(self, d: date) -> bool:
        """True if Friday or Saturday (NEPSE weekend)."""
        return d.weekday() in self.CLOSED_WEEKDAYS

    def is_public_holiday(self, d: date) -> bool:
        """True if d is a NRB-declared public holiday."""
        return d.isoformat() in self._holiday_set

    def is_bandh(self, d: date) -> bool:
        """True if a Nepal Bandh is active on d."""
        return detect_bandh_from_news(d, fetch_live=self.fetch_live)

    def is_trading_day(self, d: date) -> bool:
        """Full trading day check: not weekend, not holiday, not bandh."""
        if self.is_weekend(d):
            return False
        if self.is_public_holiday(d):
            return False
        # Bandh check only for recent/future dates (performance)
        if abs((d - date.today()).days) <= 3:
            if self.is_bandh(d):
                return False
        return True

    def next_trading_date(self, from_date: date) -> date:
        """Return the next trading day after `from_date`."""
        d = from_date + timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(d):
                return d
            d += timedelta(days=1)
        raise RuntimeError("Could not find next trading day within 30 days.")

    def next_n_trading_dates(self, from_date: date, n: int) -> List[date]:
        """Return list of next n trading dates after (not including) from_date."""
        result = []
        d = from_date
        while len(result) < n:
            d = d + timedelta(days=1)
            if self.is_trading_day(d):
                result.append(d)
            if (d - from_date).days > 60:
                break
        return result

    def prev_trading_date(self, from_date: date) -> date:
        """Return the most recent trading day before `from_date`."""
        d = from_date - timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(d):
                return d
            d -= timedelta(days=1)
        raise RuntimeError("Could not find previous trading day.")

    def trading_days_between(self, start: date, end: date) -> int:
        """Count trading days in [start, end] inclusive."""
        count = 0
        d = start
        while d <= end:
            if self.is_trading_day(d):
                count += 1
            d += timedelta(days=1)
        return count

    def days_to_next_holiday(self, from_date: date) -> int:
        """Calendar days until the next known public holiday."""
        d = from_date + timedelta(days=1)
        for _ in range(366):
            if self.is_public_holiday(d):
                return (d - from_date).days
            d += timedelta(days=1)
        return 366  # no holiday found in next year

    def get_nepali_features(self, d: date) -> Dict[str, float]:
        """
        Generate 28 Nepal-aware calendar features for ML model.
        All numeric, suitable for direct use in feature matrix.
        """
        bs_year, bs_month, bs_day = ad_to_bs(d)
        bs_days_in_month = _BS_MONTH_DATA.get(bs_year, [30]*12)[bs_month - 1]
        days_to_hol = self.days_to_next_holiday(d)

        # Fiscal year: Nepal fiscal year starts Shrawan (BS month 4 = ~July AD)
        # Q1=Shrawan-Ashwin (4-6), Q2=Kartik-Poush (7-9), Q3=Magh-Chaitra (10-12), Q4=Baisakh-Ashadh (1-3)
        fiscal_month = ((bs_month - 4) % 12) + 1  # 1=Shrawan, 12=Ashadh
        fiscal_quarter = (fiscal_month - 1) // 3 + 1

        # Festival proximity (score 0-1, peaks within 3 days of major festivals)
        festival_score = self._festival_proximity_score(d)

        # Dashain/Tihar special flag (Oct–Nov, highest market impact)
        in_dashain_tihar = float(d.month in (10, 11))

        # New Year proximity (BS New Year = Baisakh 1, ~mid April AD)
        in_new_year_window = float(bs_month == 1 and bs_day <= 7)

        # Dividend season (Ashwin/Kartik = Sep/Oct, companies announce results)
        in_dividend_season = float(bs_month in (6, 7))

        # AGM season (Baisakh–Ashadh, Q4 result announcements)
        in_agm_season = float(bs_month in (1, 2, 3))

        # NRB monetary policy (usually released Shrawan/Magh)
        nrb_policy_month = float(bs_month in (4, 10))

        # Python weekday: 0=Mon ... 4=Fri, 5=Sat, 6=Sun
        # NEPSE weekday: Sun=1st day, Thu=5th day of week
        nepse_day_of_week = {6: 1, 0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7}[d.weekday()]

        is_trading = float(self.is_trading_day(d))
        is_pre_holiday = float(
            not self.is_trading_day(d + timedelta(days=1)) and self.is_trading_day(d)
        )
        is_post_holiday = float(
            not self.is_trading_day(d - timedelta(days=1)) and self.is_trading_day(d)
        )

        return {
            # BS date components
            "bs_year":               float(bs_year),
            "bs_month":              float(bs_month),
            "bs_day":                float(bs_day),
            "bs_day_of_month_norm":  float(bs_day) / float(bs_days_in_month),
            "bs_month_name_idx":     float(((bs_month - 1) % 12) + 1),
            # Fiscal calendar
            "fiscal_month":          float(fiscal_month),
            "fiscal_quarter":        float(fiscal_quarter),
            "is_fiscal_q1":          float(fiscal_quarter == 1),
            "is_fiscal_q2":          float(fiscal_quarter == 2),
            "is_fiscal_q3":          float(fiscal_quarter == 3),
            "is_fiscal_q4":          float(fiscal_quarter == 4),
            "is_fiscal_month_start": float(bs_day <= 5),
            "is_fiscal_month_end":   float(bs_day >= bs_days_in_month - 4),
            # NEPSE session
            "nepse_day_of_week":     float(nepse_day_of_week),
            "is_trading_day":        is_trading,
            "is_pre_holiday":        is_pre_holiday,
            "is_post_holiday":       is_post_holiday,
            "days_to_next_holiday":  float(min(days_to_hol, 30)) / 30.0,
            # Festival & seasonal effects
            "festival_proximity":    festival_score,
            "in_dashain_tihar":      in_dashain_tihar,
            "in_new_year_window":    in_new_year_window,
            "in_dividend_season":    in_dividend_season,
            "in_agm_season":         in_agm_season,
            "nrb_policy_month":      nrb_policy_month,
            # AD calendar
            "ad_month":              float(d.month),
            "ad_quarter":            float((d.month - 1) // 3 + 1),
            "ad_day_of_year_norm":   float(d.timetuple().tm_yday) / 365.0,
            # Week position
            "is_week_start":         float(d.weekday() == 6),  # Sunday = first NEPSE day
            "is_week_end":           float(d.weekday() == 3),  # Thursday = last NEPSE day
        }

    def _festival_proximity_score(self, d: date) -> float:
        """
        Returns 0.0–1.0 score based on proximity to known festival dates.
        Score = 1.0 on festival day, decays by 0.25 per day out.
        """
        score = 0.0
        all_dates_str = list(self._holiday_set)
        for ds in all_dates_str:
            try:
                hd = date.fromisoformat(ds)
                delta = abs((d - hd).days)
                if delta <= 4:
                    score = max(score, 1.0 - delta * 0.25)
            except Exception:
                continue
        return round(score, 3)

    def get_holiday_name(self, d: date) -> Optional[str]:
        """Return holiday name for a date, if known."""
        _HOLIDAY_NAMES: Dict[str, str] = {
            "2025-04-14": "Nepali New Year (BS 2082)",
            "2025-05-12": "Buddha Jayanti",
            "2025-07-05": "Ashad 15 (National Paddy Day)",
            "2025-08-09": "Janai Purnima",
            "2025-08-15": "Gai Jatra",
            "2025-09-06": "Indra Jatra",
            "2025-09-22": "Dashain (Ghatasthapana)",
            "2025-10-01": "Dashain (Maha Ashtami)",
            "2025-10-02": "Dashain (Maha Navami)",
            "2025-10-03": "Vijaya Dashami",
            "2025-10-04": "Dashain (Ekadashi)",
            "2025-10-20": "Tihar (Laxmi Puja)",
            "2025-10-21": "Tihar (Gobardhan Puja)",
            "2025-10-22": "Bhai Tika",
            "2025-10-27": "Chhath Parva",
            "2026-01-15": "Maghe Sankranti",
            "2026-02-19": "National Democracy Day",
            "2026-02-26": "Maha Shivaratri",
            "2026-03-03": "Holi (Fagu Purnima)",
            "2026-04-14": "Nepali New Year (BS 2083)",
        }
        return _HOLIDAY_NAMES.get(d.isoformat())

    def upcoming_holidays(self, from_date: date, n: int = 5) -> List[Dict]:
        """Return list of upcoming n holidays from from_date."""
        result = []
        seen = set()
        d = from_date
        for _ in range(365):
            d += timedelta(days=1)
            ds = d.isoformat()
            if ds in self._holiday_set and ds not in seen:
                seen.add(ds)
                result.append({
                    "date": ds,
                    "name": self.get_holiday_name(d) or "Public Holiday",
                    "days_away": (d - from_date).days,
                })
                if len(result) >= n:
                    break
        return result


# ─── Convenience Exports ──────────────────────────────────────────────────────

_default_calendar: Optional[NepalMarketCalendar] = None


def get_calendar(fetch_live: bool = True) -> NepalMarketCalendar:
    """Get or create a singleton NepalMarketCalendar instance."""
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = NepalMarketCalendar(fetch_live=fetch_live)
    return _default_calendar


def is_nepse_trading_day(d: date, fetch_live: bool = False) -> bool:
    """Quick check: is `d` a NEPSE trading day?"""
    return get_calendar(fetch_live=fetch_live).is_trading_day(d)


def next_nepse_trading_dates(from_date: date, n: int = 7) -> List[date]:
    """Get next n NEPSE trading dates."""
    return get_calendar(fetch_live=False).next_n_trading_dates(from_date, n)


if __name__ == "__main__":
    from datetime import date
    today = date.today()
    bs = ad_to_bs(today)
    print(f"Today AD: {today} → BS: {bs[0]}-{bs[1]:02d}-{bs[2]:02d} ({get_bs_month_name(bs[1])})")

    cal = NepalMarketCalendar(fetch_live=False)
    print(f"Is trading day: {cal.is_trading_day(today)}")
    print(f"Days to next holiday: {cal.days_to_next_holiday(today)}")
    print(f"Next 5 trading days: {cal.next_n_trading_dates(today, 5)}")
    print(f"Upcoming holidays: {cal.upcoming_holidays(today, 3)}")
    feats = cal.get_nepali_features(today)
    print(f"Calendar features ({len(feats)}): {feats}")