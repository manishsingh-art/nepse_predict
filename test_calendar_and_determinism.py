from __future__ import annotations

from datetime import date

from nepse_date_utils import ad_to_bs_ymd, validate_ad_bs_mapping
from nepse_live import _determine_market_status
from nepse_market_calendar import NepseMarketCalendar
from prediction_engine import set_global_determinism


def test_weekend_rules():
    cal = NepseMarketCalendar(api_url=None, enable_bandh_detection=False)
    # 2026-03-27 is Friday, 2026-03-28 is Saturday, 2026-03-29 is Sunday
    assert cal.is_trading_day(date(2026, 3, 27)) is False
    assert cal.is_trading_day(date(2026, 3, 28)) is False
    assert cal.is_trading_day(date(2026, 3, 29)) is True


def test_ad_bs_roundtrip():
    for d in [date(2026, 3, 31), date(2026, 4, 1), date(2025, 10, 20)]:
        validate_ad_bs_mapping(d)
    assert ad_to_bs_ymd(date(2026, 4, 1)) == (2082, 12, 18)


def test_market_status_prefers_live_data():
    status = _determine_market_status(date(2026, 4, 1), True, "nepalstock_active_securities")
    assert status["label"] == "OPEN"


def test_market_status_avoids_static_holiday_close():
    status = _determine_market_status(date(2026, 4, 1), False, "live_feed_unavailable")
    assert status["label"] == "LIKELY OPEN"


def test_determinism_seed():
    set_global_determinism(123)
    # deterministic numpy
    import numpy as np

    a = np.random.RandomState(123).randn(5).tolist()
    set_global_determinism(123)
    b = np.random.RandomState(123).randn(5).tolist()
    assert a == b

