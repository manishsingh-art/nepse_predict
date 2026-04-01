from __future__ import annotations

"""
forecast_adjustment.py
----------------------
Simple forecast "conservatism" scaler to reduce overshoot in:
- high volatility regimes
- high manipulation/trap-score regimes

Tune these thresholds by backtesting.
"""


def adjust_forecast(pred: float, vol_pct: float, trap_index: float) -> float:
    """
    Adjust predicted price based on volatility and anomaly/manipulation risk.

    - vol_pct: ATR% or daily volatility percentage (e.g., 3.2 means 3.2%)
    - trap_index: 0..100
    """
    try:
        p = float(pred)
        v = float(vol_pct)
        t = float(trap_index)
    except Exception:
        return pred

    if v > 5.0 or t > 70.0:
        return p * 0.95
    return p

