from __future__ import annotations

"""
nepse_market_calendar.py
------------------------
Authoritative NEPSE market calendar facade.

Design goals:
- Correct NEPSE trading weekdays (Sun–Thu), closed Fri/Sat.
- Incorporate official holidays (external API if configured), cached locally.
- Layer in project static holiday engine (`nepal_calendar.py`) as fallback.
- Allow manual overrides for unexpected closures/re-openings.
- Provide reasons (for debug/printing) + pre-holiday low-liquidity flags.

External API:
- Optional: set `NEPSE_HOLIDAY_API_URL` (env var) to a JSON endpoint.
  Expected formats (any one):
    - {"holidays": [{"date": "YYYY-MM-DD", "name": "..."}]}
    - {"dates": ["YYYY-MM-DD", ...]}
    - [{"date": "YYYY-MM-DD", "name": "..."}]
  The module will cache the parsed result and fall back gracefully.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from nepal_calendar import NepalMarketCalendar

    _STATIC_CAL = NepalMarketCalendar(fetch_live=True)
    _HAS_STATIC = True
except Exception:
    _STATIC_CAL = None
    _HAS_STATIC = False


CACHE_DIR = Path(os.path.expanduser("~")) / ".nepse_cache" / "holidays"
DEFAULT_OVERRIDES_CSV = "nepse_holidays_overrides.csv"


@dataclass(frozen=True)
class MarketStatus:
    ad_date: date
    is_trading_day: bool
    reason: str
    holiday_name: Optional[str] = None
    warning: Optional[str] = None
    is_pre_holiday: bool = False
    pre_holiday_liquidity_flag: float = 0.0


def _to_date(d: date | datetime) -> date:
    return d.date() if isinstance(d, datetime) else d


def _read_overrides_csv(path: str | Path) -> Dict[date, Dict[str, Any]]:
    """
    Overrides CSV (optional):
      - ad_date (YYYY-MM-DD)
      - action: CLOSE | OPEN
      - reason (optional)

    CLOSE forces closure on a date (even if normally trading).
    OPEN forces trading on a date (rare; use carefully).
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        import pandas as pd

        df = pd.read_csv(p)
        if "ad_date" not in df.columns or "action" not in df.columns:
            return {}
        out: Dict[date, Dict[str, Any]] = {}
        for _, r in df.iterrows():
            try:
                ds = str(r["ad_date"]).strip()
                d = date.fromisoformat(ds)
            except Exception:
                continue
            action = str(r.get("action", "")).strip().upper()
            reason = str(r.get("reason", "")).strip() if "reason" in df.columns else ""
            if action not in ("CLOSE", "OPEN"):
                continue
            out[d] = {"action": action, "reason": reason}
        return out
    except Exception:
        return {}


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{key}.json"


def _load_cache(path: Path, ttl_seconds: int) -> Optional[dict]:
    try:
        if not path.exists():
            return None
        age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
        if age > ttl_seconds:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(path: Path, payload: dict) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass


def _parse_api_holidays(payload: Any) -> Dict[str, str]:
    """
    Returns mapping date_str -> name (name may be empty).
    """
    out: Dict[str, str] = {}
    if payload is None:
        return out

    # {"holidays":[{"date":"YYYY-MM-DD","name":"..."}]}
    if isinstance(payload, dict) and isinstance(payload.get("holidays"), list):
        for h in payload["holidays"]:
            if isinstance(h, dict) and "date" in h:
                ds = str(h["date"]).strip()
                name = str(h.get("name") or h.get("reason") or "").strip()
                out[ds] = name
        return out

    # {"dates":["YYYY-MM-DD", ...]}
    if isinstance(payload, dict) and isinstance(payload.get("dates"), list):
        for ds in payload["dates"]:
            ds2 = str(ds).strip()
            if ds2:
                out[ds2] = ""
        return out

    # [{"date":"YYYY-MM-DD","name":"..."}] or ["YYYY-MM-DD", ...]
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, str):
                ds = item.strip()
                if ds:
                    out[ds] = ""
            elif isinstance(item, dict) and "date" in item:
                ds = str(item["date"]).strip()
                name = str(item.get("name") or item.get("reason") or "").strip()
                if ds:
                    out[ds] = name
    return out


def _fetch_official_holidays(url: str, timeout_s: int = 10) -> Optional[Dict[str, str]]:
    try:
        import requests  # optional dependency at runtime

        r = requests.get(url, timeout=timeout_s, headers={"User-Agent": "nepse_predict/1.0"})
        if r.status_code != 200:
            return None
        return _parse_api_holidays(r.json())
    except Exception:
        return None


class NepseMarketCalendar:
    def __init__(
        self,
        overrides_csv: str | Path = DEFAULT_OVERRIDES_CSV,
        api_url: Optional[str] = None,
        api_cache_ttl_seconds: int = 3600 * 24 * 7,
        enable_bandh_detection: bool = True,
    ):
        self.overrides_csv = overrides_csv
        self.api_url = api_url or os.getenv("NEPSE_HOLIDAY_API_URL")
        self.api_cache_ttl_seconds = int(api_cache_ttl_seconds)
        self.enable_bandh_detection = bool(enable_bandh_detection)

        self._overrides = _read_overrides_csv(self.overrides_csv)
        self._api_holidays = self._load_api_holidays()

    # --- Holiday sources ---
    def _load_api_holidays(self) -> Dict[str, str]:
        if not self.api_url:
            return {}

        key = f"official_{abs(hash(self.api_url))}"
        path = _cache_path(key)
        cached = _load_cache(path, ttl_seconds=self.api_cache_ttl_seconds)
        if isinstance(cached, dict) and isinstance(cached.get("holidays"), dict):
            return {str(k): str(v) for k, v in cached["holidays"].items()}

        holidays = _fetch_official_holidays(self.api_url)
        if holidays is None:
            return {}

        _save_cache(path, {"fetched_at": datetime.now().isoformat(timespec="seconds"), "holidays": holidays})
        return holidays

    def _api_is_holiday(self, d: date) -> Tuple[bool, Optional[str]]:
        ds = d.isoformat()
        if ds in self._api_holidays:
            name = self._api_holidays.get(ds) or None
            return True, name
        return False, None

    # --- Core logic ---

    def _is_non_trading_basic(self, dd: date) -> bool:
        """
        Non-recursive closure check: weekend + overrides + API + static public holidays only.
        Does NOT call is_trading_day / market_status / is_pre_holiday — safe to use inside
        those methods without creating a cycle.
        """
        ov = self._overrides.get(dd)
        if ov:
            return ov["action"] == "CLOSE"
        if dd.weekday() in (4, 5):
            return True
        is_h, _ = self._api_is_holiday(dd)
        if is_h:
            return True
        if _HAS_STATIC and _STATIC_CAL is not None:
            try:
                if getattr(_STATIC_CAL, "is_public_holiday", None) and _STATIC_CAL.is_public_holiday(dd):
                    return True
            except Exception:
                pass
        return False

    def _compute_pre_holiday(self, dd: date, lookahead_days: int = 1) -> bool:
        """
        Internal: returns True when one of the next `lookahead_days` calendar days is
        non-trading. Uses _is_non_trading_basic so there is no recursion risk.
        """
        for i in range(1, max(1, int(lookahead_days)) + 1):
            if self._is_non_trading_basic(dd + timedelta(days=i)):
                return True
        return False

    def market_status(self, d: date | datetime) -> MarketStatus:
        dd = _to_date(d)

        # Manual overrides win
        ov = self._overrides.get(dd)
        if ov:
            if ov["action"] == "CLOSE":
                return MarketStatus(dd, False, reason=ov.get("reason") or "Manual override: CLOSED")
            if ov["action"] == "OPEN":
                return MarketStatus(dd, True, reason=ov.get("reason") or "Manual override: OPEN")

        # Weekend (NEPSE): Friday=4, Saturday=5
        if dd.weekday() in (4, 5):
            return MarketStatus(dd, False, reason="Weekend (Fri/Sat closed)")

        # Official holidays (API)
        is_h, name = self._api_is_holiday(dd)
        if is_h:
            return MarketStatus(dd, False, reason="Official holiday (API)", holiday_name=name)

        # Static engine fallback (includes known holiday set, and bandh detection if enabled)
        if _HAS_STATIC and _STATIC_CAL is not None:
            if not self.enable_bandh_detection and hasattr(_STATIC_CAL, "is_weekend"):
                if _STATIC_CAL.is_public_holiday(dd):
                    return MarketStatus(dd, False, reason="Public holiday (static)", holiday_name=_STATIC_CAL.get_holiday_name(dd))
            else:
                if not _STATIC_CAL.is_trading_day(dd):
                    if getattr(_STATIC_CAL, "is_public_holiday", None) and _STATIC_CAL.is_public_holiday(dd):
                        return MarketStatus(dd, False, reason="Public holiday (static)", holiday_name=_STATIC_CAL.get_holiday_name(dd))
                    if getattr(_STATIC_CAL, "is_bandh", None) and _STATIC_CAL.is_bandh(dd):
                        return MarketStatus(dd, False, reason="Bandh/strike detected")
                    return MarketStatus(dd, False, reason="Market closed (static rules)")
                # If trading day but bandh is uncertain, flag it without closing.
                try:
                    if hasattr(_STATIC_CAL, "bandh_status"):
                        bs = _STATIC_CAL.bandh_status(dd)
                        if bool(bs.get("uncertain", False)):
                            is_pre = self._compute_pre_holiday(dd)
                            return MarketStatus(
                                dd,
                                True,
                                reason="Trading day",
                                warning=f"⚠ possible disruption (bandh/strike unconfirmed, score={bs.get('score')}, sources={bs.get('sources')})",
                                is_pre_holiday=is_pre,
                                pre_holiday_liquidity_flag=1.0 if is_pre else 0.0,
                            )
                except Exception:
                    pass

        # Otherwise: trading day — use non-recursive pre-holiday check
        is_pre = self._compute_pre_holiday(dd)
        return MarketStatus(
            dd,
            True,
            reason="Trading day",
            is_pre_holiday=is_pre,
            pre_holiday_liquidity_flag=1.0 if is_pre else 0.0,
        )

    def is_trading_day(self, d: date | datetime) -> bool:
        return self.market_status(d).is_trading_day

    def next_trading_day(self, from_date: date | datetime) -> date:
        dd = _to_date(from_date)
        d = dd + timedelta(days=1)
        for _ in range(60):
            if self.is_trading_day(d):
                return d
            d += timedelta(days=1)
        raise RuntimeError("Could not find next trading day within 60 days.")

    def next_n_trading_days(self, from_date: date | datetime, n: int) -> List[date]:
        dd = _to_date(from_date)
        out: List[date] = []
        d = dd
        while len(out) < int(n):
            d = d + timedelta(days=1)
            if self.is_trading_day(d):
                out.append(d)
            if (d - dd).days > 120:
                break
        return out

    def is_pre_holiday(self, d: date | datetime, lookahead_days: int = 1) -> bool:
        """
        Pre-holiday: a trading day whose next calendar day(s) contains a non-trading day.
        This is a practical thin-liquidity flag for NEPSE.
        """
        dd = _to_date(d)
        if self._is_non_trading_basic(dd):
            return False
        return self._compute_pre_holiday(dd, lookahead_days)

    def upcoming_holidays(self, from_date: date | datetime, n: int = 3) -> List[Dict[str, Any]]:
        """
        Upcoming known public holidays (API + static), excluding weekends.
        Best-effort and primarily for UI display.
        """
        dd = _to_date(from_date)
        results: List[Dict[str, Any]] = []
        seen: set[str] = set()
        d = dd
        for _ in range(366):
            d += timedelta(days=1)
            ds = d.isoformat()
            if ds in seen:
                continue
            if d.weekday() in (4, 5):
                continue

            api_is, api_name = self._api_is_holiday(d)
            static_is = False
            static_name = None
            if _HAS_STATIC and _STATIC_CAL is not None:
                try:
                    static_is = bool(_STATIC_CAL.is_public_holiday(d))
                    static_name = _STATIC_CAL.get_holiday_name(d) if static_is else None
                except Exception:
                    static_is = False

            if api_is or static_is:
                seen.add(ds)
                results.append(
                    {"date": ds, "name": api_name or static_name or "Public holiday", "days_away": (d - dd).days}
                )
                if len(results) >= int(n):
                    break
        return results


def get_market_calendar() -> NepseMarketCalendar:
    """
    Singleton-ish convenience for most call sites.
    """
    # keep a module-level singleton without global mutation complexity
    if not hasattr(get_market_calendar, "_inst"):
        setattr(get_market_calendar, "_inst", NepseMarketCalendar())
    return getattr(get_market_calendar, "_inst")


def validate_trading_date_alignment(ad_dates: Iterable[date]) -> None:
    """
    Validation helper: ensures all dates are trading days and monotonic.
    """
    cal = get_market_calendar()
    prev: Optional[date] = None
    for d in ad_dates:
        if not cal.is_trading_day(d):
            st = cal.market_status(d)
            raise AssertionError(f"Non-trading date in trading sequence: {d} ({st.reason})")
        if prev and d <= prev:
            raise AssertionError(f"Non-increasing trading date sequence: {prev} -> {d}")
        prev = d

