from __future__ import annotations

"""
microstructure.py
-----------------
Guardrails for illiquid/missing floorsheet conditions.
"""

from typing import Any


def safe_forecast(floorsheet_df: Any, predicted_price: float) -> float:
    """
    Adjust prediction if floorsheet data is missing or illiquid.

    Expected floorsheet_df to behave like a pandas DataFrame with a 'volume' column,
    but this function is defensive and will degrade gracefully.
    """
    try:
        if floorsheet_df is None:
            return float(predicted_price) * 0.97
        if getattr(floorsheet_df, "empty", False):
            return float(predicted_price) * 0.97
        if "volume" in getattr(floorsheet_df, "columns", []):
            vol_sum = float(floorsheet_df["volume"].sum())
            if vol_sum < 1000:
                return float(predicted_price) * 0.97
    except Exception:
        return float(predicted_price) * 0.97
    return float(predicted_price)

