"""
Central NEPSE regulatory and microstructure parameters.

Defaults match **NEPSE Predictor v6.4** behaviour. Override via environment
variables (see ``get_effective_rules``) or by editing this module — model cache
keys include ``rules_fingerprint()`` so cached ensembles invalidate when rules
change.

**Assumptions (verify against current SEBON / NEPSE circulars):**

- Single-session price band is modelled as symmetric ``±CIRCUIT_BREAKER_PCT``
  (commonly 10% for NEPSE equities; adjust if policy differs).
- ``FEE_RATE`` is a **single round-trip friction** used by the backtest engine
  (broker, SEBON, CDS, etc. blended — not tax advice).
- ``TRADING_SESSIONS_PER_YEAR`` defaults to **252** for Sharpe annualization
  (same as v6.4). NEPSE has fewer physical sessions; set
  ``NEPSE_TRADING_SESSIONS_PER_YEAR`` (e.g. 250) for stricter realism.
- Settlement label ``T+2`` is informational for reports; settlement is not
  simulated bar-by-bar in this codebase.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from typing import Any, Dict, Tuple, Union

import numpy as np

RULES_VERSION = "2026.04.12"

# Published defaults (v6.4 baseline). Mutable copy returned by get_effective_rules().
NEPSE_RULES: Dict[str, Any] = {
    "RULES_VERSION": RULES_VERSION,
    # Price band per regular session (fraction, e.g. 0.10 = ±10%)
    "CIRCUIT_BREAKER_PCT": 0.10,
    # Human-readable; calendar code uses WEEKEND_WEEKDAYS (Python weekday ints).
    "TRADING_DAYS": ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"],
    # Python ``date.weekday()``: Monday=0 … Sunday=6. NEPSE closed Friday & Saturday.
    "WEEKEND_WEEKDAYS": [4, 5],
    # Nepal Standard Time — informational (not used for bar timestamps in v6.4).
    "TRADING_HOURS": ("11:00", "15:00"),
    # Backtest: one combined round-trip cost rate (v6.4 default).
    "FEE_RATE": 0.004,
    "SLIPPAGE_RATE": 0.001,
    "SETTLEMENT_CYCLE": "T+2",
    "SHORT_SELLING_ALLOWED": False,
    # Sharpe annualization (√N). v6.4 used 252.
    "TRADING_SESSIONS_PER_YEAR": 252,
    # decision_engine: scale expected multi-day move and ATR% penalties.
    "DECISION_EXPECTED_RET_REF_PCT": 8.0,
    "DECISION_VOLATILITY_REF_PCT": 8.0,
    # features: volume quantile below which a bar is flagged illiquid.
    "ILLIQUID_VOLUME_QUANTILE": 0.20,
    "PRICE_RANK_LOOKBACK_BARS": 252,
    # regime.py thresholds (ATR% and volume ratios).
    "REGIME_VOL_BULL_HEALTHY_MAX": 3.5,
    "REGIME_VOL_BULL_PARABOLIC_MIN": 5.5,
    "REGIME_VOL_BEAR_SLOW_MAX": 2.5,
    "REGIME_VOL_SIDEWAYS_MAX": 1.5,
    "REGIME_VOL_RATIO_SIDEWAYS_MAX": 0.8,
    "REGIME_TRAP_MANIPULATION": 60,
    "REGIME_TRAP_BULL_EXHAUST": 50,
    "REGIME_BUY_HHI_QUIET": 2000,
}


def _parse_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        warnings.warn(
            f"Invalid {key}={raw!r}; using default {default}",
            UserWarning,
            stacklevel=3,
        )
        return float(default)


def _parse_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        warnings.warn(
            f"Invalid {key}={raw!r}; using default {default}",
            UserWarning,
            stacklevel=3,
        )
        return int(default)


def _warn_old_env_detected() -> None:
    """Emit once if deprecated / legacy env hints are present."""
    if os.environ.get("NEPSE_ASSUME_LEGACY_10PCT_ONLY") == "1":
        warnings.warn(
            "NEPSE_ASSUME_LEGACY_10PCT_ONLY=1: circuit band still follows "
            "NEPSE_CIRCUIT_BREAKER_PCT / CIRCUIT_BREAKER_PCT in config — "
            "unset this after migrating to config/nepse_rules.py.",
            UserWarning,
            stacklevel=3,
        )


def get_effective_rules() -> Dict[str, Any]:
    """
    Return a fresh dict of effective rules with environment overrides applied.

    Environment variables (all optional):
        NEPSE_CIRCUIT_BREAKER_PCT
        NEPSE_FEE_RATE
        NEPSE_SLIPPAGE_RATE
        NEPSE_TRADING_SESSIONS_PER_YEAR
        NEPSE_DECISION_EXPECTED_RET_REF_PCT
        NEPSE_DECISION_VOLATILITY_REF_PCT
        NEPSE_ILLIQUID_VOLUME_QUANTILE
        NEPSE_PRICE_RANK_LOOKBACK_BARS
        NEPSE_USE_LEGACY_V64 — if ``1``, skip env overrides (reproducibility).
    """
    use_legacy = os.environ.get("NEPSE_USE_LEGACY_V64", "").strip() in ("1", "true", "TRUE", "yes", "YES")
    if use_legacy:
        warnings.warn(
            "NEPSE_USE_LEGACY_V64 active: environment overrides to nepse_rules are ignored.",
            UserWarning,
            stacklevel=2,
        )
        return dict(NEPSE_RULES)

    _warn_old_env_detected()

    r = dict(NEPSE_RULES)
    r["CIRCUIT_BREAKER_PCT"] = _parse_float("NEPSE_CIRCUIT_BREAKER_PCT", float(r["CIRCUIT_BREAKER_PCT"]))
    r["FEE_RATE"] = _parse_float("NEPSE_FEE_RATE", float(r["FEE_RATE"]))
    r["SLIPPAGE_RATE"] = _parse_float("NEPSE_SLIPPAGE_RATE", float(r["SLIPPAGE_RATE"]))
    r["TRADING_SESSIONS_PER_YEAR"] = _parse_int(
        "NEPSE_TRADING_SESSIONS_PER_YEAR", int(r["TRADING_SESSIONS_PER_YEAR"])
    )
    r["DECISION_EXPECTED_RET_REF_PCT"] = _parse_float(
        "NEPSE_DECISION_EXPECTED_RET_REF_PCT", float(r["DECISION_EXPECTED_RET_REF_PCT"])
    )
    r["DECISION_VOLATILITY_REF_PCT"] = _parse_float(
        "NEPSE_DECISION_VOLATILITY_REF_PCT", float(r["DECISION_VOLATILITY_REF_PCT"])
    )
    r["ILLIQUID_VOLUME_QUANTILE"] = _parse_float(
        "NEPSE_ILLIQUID_VOLUME_QUANTILE", float(r["ILLIQUID_VOLUME_QUANTILE"])
    )
    r["PRICE_RANK_LOOKBACK_BARS"] = _parse_int(
        "NEPSE_PRICE_RANK_LOOKBACK_BARS", int(r["PRICE_RANK_LOOKBACK_BARS"])
    )
    r["RULES_VERSION"] = RULES_VERSION
    return r


def rules_fingerprint() -> str:
    """Short hash of effective rules for cache keys (invalidates when rules change)."""
    eff = get_effective_rules()
    payload = json.dumps(eff, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def weekend_weekdays() -> Tuple[int, ...]:
    """Weekdays (Python convention) on which NEPSE is closed."""
    w = get_effective_rules().get("WEEKEND_WEEKDAYS", [4, 5])
    if isinstance(w, (list, tuple)):
        return tuple(int(x) for x in w)
    return (4, 5)


def circuit_price_bounds(
    prev_price: float,
    circuit_pct: Union[float, None] = None,
) -> Tuple[float, float]:
    pct = float(circuit_pct if circuit_pct is not None else get_effective_rules()["CIRCUIT_BREAKER_PCT"])
    lo = float(prev_price) * (1.0 - pct)
    hi = float(prev_price) * (1.0 + pct)
    return lo, hi


def clip_circuit_price(
    price: float,
    prev_price: float,
    circuit_pct: Union[float, None] = None,
) -> float:
    lo, hi = circuit_price_bounds(prev_price, circuit_pct)
    return float(np.clip(price, lo, hi))


def annualized_sharpe_factor() -> float:
    """sqrt(N) for N trading sessions per year (used with mean/std of daily returns)."""
    n = int(get_effective_rules()["TRADING_SESSIONS_PER_YEAR"])
    return float(math.sqrt(max(n, 1)))
