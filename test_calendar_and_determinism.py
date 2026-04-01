from __future__ import annotations

from datetime import date

from nepse_market_calendar import NepseMarketCalendar
from prediction_engine import set_global_determinism
from nepse_date_utils import validate_ad_bs_mapping


def test_weekend_rules():
    cal = NepseMarketCalendar(api_url=None, enable_bandh_detection=False)
    # 2026-03-27 is Friday, 2026-03-28 is Saturday, 2026-03-29 is Sunday
    assert cal.is_trading_day(date(2026, 3, 27)) is False
    assert cal.is_trading_day(date(2026, 3, 28)) is False
    assert cal.is_trading_day(date(2026, 3, 29)) is True


def test_ad_bs_roundtrip():
    for d in [date(2026, 3, 31), date(2026, 4, 1), date(2025, 10, 20)]:
        validate_ad_bs_mapping(d)


def test_determinism_seed():
    set_global_determinism(123)
    # deterministic numpy
    import numpy as np

    a = np.random.RandomState(123).randn(5).tolist()
    set_global_determinism(123)
    b = np.random.RandomState(123).randn(5).tolist()
    assert a == b

