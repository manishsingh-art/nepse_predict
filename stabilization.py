from __future__ import annotations

"""
stabilization.py
----------------
Post-processing / guardrails for NEPSE forecasts in low-liquidity conditions.

Applied per-step (recursive forecast):
- ATR/volatility-based daily move caps
- Mean-reversion force (z-score to SMA)
- Support/resistance gravity
- Volatility decay over horizon
- Deterministic exponential smoothing
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StabilizationParams:
    k_atr: float = 1.25              # cap = k_atr * ATR%
    vol_cap_percentile: float = 0.90 # cap by historical vol percentile (0..1)
    mean_reversion_strength: float = 0.18
    sr_gravity_strength: float = 0.25
    horizon_vol_decay: float = 0.06  # tighter caps as horizon increases
    smoothing_alpha: float = 0.35


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def estimate_hist_vol_cap(hist_rets: np.ndarray, percentile: float) -> float:
    """
    Returns a volatility cap in fractional terms (e.g. 0.03 for 3%).
    Uses absolute daily returns distribution.
    """
    if hist_rets.size < 10:
        return 0.08
    absr = np.abs(hist_rets[np.isfinite(hist_rets)])
    if absr.size < 10:
        return 0.08
    p = float(np.clip(percentile, 0.5, 0.99))
    return float(np.quantile(absr, p))


def cap_daily_move(prev_price: float, raw_price: float, cap_frac: float) -> float:
    p0 = float(prev_price)
    p1 = float(raw_price)
    cap = abs(float(cap_frac))
    lo = p0 * (1.0 - cap)
    hi = p0 * (1.0 + cap)
    return float(np.clip(p1, lo, hi))


def apply_mean_reversion(prev_price: float, price: float, feats: Dict[str, Any], strength: float) -> float:
    """
    If z-score is extreme, pull price toward prev_price (cooling effect).
    """
    z = _safe_float(feats.get("zscore_20", 0.0), 0.0)
    s = float(np.clip(strength, 0.0, 0.8))
    if z > 2.0 and price > prev_price:
        return prev_price + (price - prev_price) * (1.0 - s)
    if z < -2.0 and price < prev_price:
        return prev_price + (price - prev_price) * (1.0 - s * 0.5)
    return price


def apply_sr_gravity(prev_price: float, price: float, feats: Dict[str, Any], strength: float) -> float:
    """
    If forecast jumps beyond resistance/support, dampen the move.
    """
    res = _safe_float(feats.get("resist_20", 0.0), 0.0)
    sup = _safe_float(feats.get("support_20", 0.0), 0.0)
    s = float(np.clip(strength, 0.0, 0.9))
    if res > 0 and price > res and price > prev_price:
        return prev_price + (price - prev_price) * (1.0 - s)
    if sup > 0 and price < sup and price < prev_price:
        return prev_price + (price - prev_price) * (1.0 - s * 0.6)
    return price


def exp_smooth(prev_smoothed: Optional[float], new_value: float, alpha: float) -> float:
    a = float(np.clip(alpha, 0.0, 1.0))
    if prev_smoothed is None:
        return float(new_value)
    return float(a * new_value + (1.0 - a) * prev_smoothed)


def stabilize_forecast_step(
    *,
    step: int,
    prev_price: float,
    raw_pred: float,
    atr_pct: float,
    hist_rets: np.ndarray,
    feats: Dict[str, Any],
    params: StabilizationParams,
    prev_smoothed: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Returns (stabilized_price, smoothed_price).
    """
    p0 = float(prev_price)
    p = float(raw_pred)

    atr_cap = float(params.k_atr) * abs(float(atr_pct))
    vol_cap = estimate_hist_vol_cap(hist_rets, params.vol_cap_percentile)

    # Tighten caps as horizon increases
    decay = 1.0 - float(params.horizon_vol_decay) * max(0, int(step) - 1)
    decay = float(np.clip(decay, 0.6, 1.0))
    cap = min(atr_cap, vol_cap) * decay
    cap = float(np.clip(cap, 0.01, 0.10))  # NEPSE realism: keep within 1%..10%

    p = cap_daily_move(p0, p, cap)
    p = apply_mean_reversion(p0, p, feats, params.mean_reversion_strength)
    p = apply_sr_gravity(p0, p, feats, params.sr_gravity_strength)

    smoothed = exp_smooth(prev_smoothed, p, params.smoothing_alpha)
    return float(p), float(smoothed)

