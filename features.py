#!/usr/bin/env python3
"""
features.py — Comprehensive Feature Engineering for NEPSE
===========================================================
Produces 80+ features across:
  - Price action (OHLCV-derived)
  - Technical indicators (trend, momentum, volatility, volume)
  - Market microstructure (spreads, efficiency)
  - Calendar effects (day-of-week, month, fiscal year)
  - Cross-asset proxies (sentiment score injection)
  - Regime detection (hidden Markov-like heuristic)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ─── Main Feature Builder ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
    """
    Adds all features to a copy of `df`.
    `sentiment_score` is a scalar [-1..+1] from news analysis.

    Returns DataFrame with all original columns + feature columns.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _ensure_ohlcv(df)

    cl = df["close"].astype(float)
    op = df["open"].astype(float)
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # ── Returns ───────────────────────────────────────────────────────────────
    for p in [1, 2, 3, 5, 7, 10, 15, 20]:
        df[f"ret_{p}d"] = cl.pct_change(p, fill_method=None)
        df[f"log_ret_{p}d"] = np.log(cl / cl.shift(p))

    # ── Moving Averages ───────────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100]:
        df[f"sma_{w}"] = cl.rolling(w).mean()
        df[f"ema_{w}"] = cl.ewm(span=w, adjust=False).mean()
        df[f"dist_sma_{w}"] = (cl - df[f"sma_{w}"]) / (df[f"sma_{w}"] + 1e-9)
        df[f"dist_ema_{w}"] = (cl - df[f"ema_{w}"]) / (df[f"ema_{w}"] + 1e-9)

    # MA crossover signals
    df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(float)
    df["sma_10_50_cross"] = (df["sma_10"] > df["sma_50"]).astype(float)
    df["ema_12_26_cross"] = (df["ema_12"] > df["ema_26"]).astype(float)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = cl.ewm(span=12, adjust=False).mean()
    ema26 = cl.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_norm"] = df["macd"] / (cl + 1e-9)

    # ── RSI family ────────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"rsi_{period}"] = _rsi(cl, period)

    # Stochastic RSI
    rsi14 = df["rsi_14"]
    rsi_min = rsi14.rolling(14).min()
    rsi_max = rsi14.rolling(14).max()
    df["stoch_rsi"] = (rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for w in [20, 50]:
        mid = cl.rolling(w).mean()
        std = cl.rolling(w).std()
        df[f"bb_upper_{w}"] = mid + 2 * std
        df[f"bb_lower_{w}"] = mid - 2 * std
        df[f"bb_width_{w}"] = (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"]) / (mid + 1e-9)
        df[f"bb_pos_{w}"] = (cl - df[f"bb_lower_{w}"]) / (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"] + 1e-9)
        # Squeeze indicator: BB width < 90th percentile
        df[f"bb_squeeze_{w}"] = (df[f"bb_width_{w}"] < df[f"bb_width_{w}"].rolling(250).quantile(0.1)).astype(float)

    # ── ATR & Volatility ──────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"atr_{period}"] = _atr(hi, lo, cl, period)
        df[f"atr_pct_{period}"] = df[f"atr_{period}"] / (cl + 1e-9)

    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_ratio"] = df["vol_10d"] / (df["vol_20d"] + 1e-9)  # vol compression signal

    # Garman-Klass volatility estimator (uses OHLC — more efficient)
    log_hl = np.log(hi / lo.replace(0, np.nan))
    log_co = np.log(cl / op.replace(0, np.nan))
    df["gk_vol"] = np.sqrt(
        0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    ).rolling(20).mean()

    # ── Stochastic Oscillator ─────────────────────────────────────────────────
    for period in [9, 14]:
        lo_n = lo.rolling(period).min()
        hi_n = hi.rolling(period).max()
        k = 100 * (cl - lo_n) / (hi_n - lo_n + 1e-9)
        df[f"stoch_k_{period}"] = k
        df[f"stoch_d_{period}"] = k.rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────────────────────
    hi14 = hi.rolling(14).max()
    lo14 = lo.rolling(14).min()
    df["williams_r"] = -100 * (hi14 - cl) / (hi14 - lo14 + 1e-9)

    # ── CCI (Commodity Channel Index) ────────────────────────────────────────
    tp = (hi + lo + cl) / 3
    tp_ma = tp.rolling(20).mean()
    tp_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp_ma) / (0.015 * tp_dev + 1e-9)

    # ── On-Balance Volume & Volume Indicators ────────────────────────────────
    obv_vals = [0.0]
    for i in range(1, len(df)):
        if cl.iloc[i] > cl.iloc[i - 1]:
            obv_vals.append(obv_vals[-1] + vol.iloc[i])
        elif cl.iloc[i] < cl.iloc[i - 1]:
            obv_vals.append(obv_vals[-1] - vol.iloc[i])
        else:
            obv_vals.append(obv_vals[-1])
    df["obv"] = obv_vals
    df["obv_ema"] = pd.Series(obv_vals).ewm(span=10, adjust=False).values
    df["obv_trend"] = (df["obv"] > df["obv_ema"]).astype(float)

    df["vol_sma_5"] = vol.rolling(5).mean()
    df["vol_sma_20"] = vol.rolling(20).mean()
    df["vol_ratio_5_20"] = df["vol_sma_5"] / (df["vol_sma_20"] + 1e-9)
    df["vol_zscore"] = (vol - df["vol_sma_20"]) / (vol.rolling(20).std() + 1e-9)
    df["vol_price_trend"] = df["ret_1d"] * df["vol_zscore"]  # price-volume confirmation

    # ── Ichimoku Cloud (simplified) ───────────────────────────────────────────
    high9  = hi.rolling(9).max()
    low9   = lo.rolling(9).min()
    high26 = hi.rolling(26).max()
    low26  = lo.rolling(26).min()
    df["ichi_conv"]    = (high9 + low9) / 2
    df["ichi_base"]    = (high26 + low26) / 2
    df["ichi_span_a"]  = ((df["ichi_conv"] + df["ichi_base"]) / 2).shift(26)
    df["ichi_span_b"]  = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    df["above_cloud"]  = (cl > df["ichi_span_a"].fillna(0)).astype(float)

    # ── Price Action Features ─────────────────────────────────────────────────
    df["body_size"]     = (cl - op).abs() / (cl + 1e-9)
    df["upper_shadow"]  = (hi - pd.concat([cl, op], axis=1).max(axis=1)) / (cl + 1e-9)
    df["lower_shadow"]  = (pd.concat([cl, op], axis=1).min(axis=1) - lo) / (cl + 1e-9)
    df["body_to_range"] = (cl - op).abs() / (hi - lo + 1e-9)
    df["gap_up"]        = ((op > cl.shift(1)) & (op > cl.shift(1) * 1.005)).astype(float)
    df["gap_down"]      = ((op < cl.shift(1)) & (op < cl.shift(1) * 0.995)).astype(float)

    # Doji detection (indecision candle)
    df["doji"] = (df["body_size"] < 0.002).astype(float)

    # Inside bar (consolidation)
    df["inside_bar"] = (
        (hi < hi.shift(1)) & (lo > lo.shift(1))
    ).astype(float)

    # ── Support / Resistance Distance ────────────────────────────────────────
    for w in [10, 20, 50]:
        df[f"support_{w}"] = lo.rolling(w).min()
        df[f"resist_{w}"]  = hi.rolling(w).max()
        df[f"dist_support_{w}"] = (cl - df[f"support_{w}"]) / (cl + 1e-9)
        df[f"dist_resist_{w}"]  = (df[f"resist_{w}"] - cl) / (cl + 1e-9)
        df[f"sr_ratio_{w}"]     = df[f"dist_support_{w}"] / (df[f"dist_resist_{w}"] + 1e-9)

    # ── Momentum & Rate of Change ─────────────────────────────────────────────
    for p in [5, 10, 20]:
        df[f"roc_{p}"] = cl.pct_change(p, fill_method=None) * 100
        df[f"momentum_{p}"] = cl - cl.shift(p)

    # Price acceleration (2nd derivative)
    df["price_accel"] = df["ret_1d"] - df["ret_1d"].shift(1)

    # ── Efficiency Ratio (Kaufman) ────────────────────────────────────────────
    for w in [10, 20]:
        direction = (cl - cl.shift(w)).abs()
        volatility_sum = cl.diff().abs().rolling(w).sum()
        df[f"efficiency_ratio_{w}"] = direction / (volatility_sum + 1e-9)

    # ── Fractal Dimension / Noise ─────────────────────────────────────────────
    # High fractal dim = choppy market, low = trending
    for w in [20]:
        n1 = (hi.rolling(w // 2).max() - lo.rolling(w // 2).min()).iloc[w // 2 - 1:]
        # simplified: use std(returns)/mean(|returns|) as noise metric
        rets = df["ret_1d"].rolling(w)
        df[f"noise_{w}"] = rets.std() / (rets.apply(lambda x: np.mean(np.abs(x)), raw=True) + 1e-9)

    # ── Calendar Features ─────────────────────────────────────────────────────
    df["dow"]              = df["date"].dt.dayofweek          # 0=Mon, 4=Fri
    df["month"]            = df["date"].dt.month
    df["quarter"]          = df["date"].dt.quarter
    df["is_month_start"]   = (df["date"].dt.day <= 5).astype(float)
    df["is_month_end"]     = (df["date"].dt.day >= 25).astype(float)
    # NEPSE fiscal year starts Shrawan (mid-July); proxy: month >= 7 or <= 3
    df["is_fiscal_q1"]     = df["month"].isin([7, 8, 9]).astype(float)
    df["is_fiscal_q4"]     = df["month"].isin([4, 5, 6]).astype(float)
    # Day-of-week dummies
    for d in range(5):
        df[f"dow_{d}"] = (df["dow"] == d).astype(float)

    # ── Lag Features ─────────────────────────────────────────────────────────
    for lag in [1, 2, 3, 4, 5, 7, 10, 15]:
        df[f"close_lag_{lag}"] = cl.shift(lag)
        df[f"ret_lag_{lag}"]   = df["ret_1d"].shift(lag)
        df[f"vol_lag_{lag}"]   = vol.shift(lag)

    # ── Z-Score Normalised Price ──────────────────────────────────────────────
    for w in [20, 50]:
        mu = cl.rolling(w).mean()
        sd = cl.rolling(w).std()
        df[f"zscore_{w}"] = (cl - mu) / (sd + 1e-9)

    # ── Trend Regime ─────────────────────────────────────────────────────────
    # Heuristic: 3-state regime (0=bear, 1=sideways, 2=bull) using SMA alignment
    df["regime"] = 1  # default sideways
    bull_mask = (cl > df["sma_20"]) & (df["sma_20"] > df["sma_50"])
    bear_mask = (cl < df["sma_20"]) & (df["sma_20"] < df["sma_50"])
    df.loc[bull_mask, "regime"] = 2
    df.loc[bear_mask, "regime"] = 0

    # Regime persistence: how many consecutive bars in same regime
    df["regime_bars"] = _streak(df["regime"])

    # ── Sentiment Injection ───────────────────────────────────────────────────
    df["sentiment_score"] = sentiment_score
    # Interaction: sentiment * momentum
    df["sentiment_x_momentum"] = sentiment_score * df["ret_5d"].fillna(0)
    df["sentiment_x_rsi"] = sentiment_score * (df["rsi_14"].fillna(50) - 50) / 50

    # ── Rolling Skewness & Kurtosis of Returns ────────────────────────────────
    df["ret_skew_20"] = df["ret_1d"].rolling(20).skew()
    df["ret_kurt_20"] = df["ret_1d"].rolling(20).kurt()

    # ── 52-week rank ──────────────────────────────────────────────────────────
    rolling_min_252 = cl.rolling(min(252, len(df))).min()
    rolling_max_252 = cl.rolling(min(252, len(df))).max()
    df["price_rank_52w"] = (cl - rolling_min_252) / (rolling_max_252 - rolling_min_252 + 1e-9)

    return df


# ── Target Engineering ────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds prediction targets:
    - target_next_close: next session close (regression)
    - target_direction: 1=up, 0=down next session (classification)
    - target_ret_1d: next session return (regression, normalised)
    - target_ret_5d: 5-day forward return (used for signal evaluation)
    """
    cl = df["close"].astype(float)
    df["target_next_close"] = cl.shift(-1)
    df["target_ret_1d"]     = cl.pct_change(fill_method=None).shift(-1)
    df["target_ret_5d"]     = cl.pct_change(5, fill_method=None).shift(-5)
    df["target_direction"]  = (df["target_ret_1d"] > 0).astype(int)
    return df


# ── Feature Selection ─────────────────────────────────────────────────────────

# Core feature set (always used) — ordered by general importance
BASE_FEATURES = [
    # Price-relative
    "dist_sma_5", "dist_sma_10", "dist_sma_20", "dist_sma_50",
    "dist_ema_5", "dist_ema_10", "dist_ema_20",
    # Returns
    "ret_1d", "ret_2d", "ret_3d", "ret_5d", "ret_10d", "ret_20d",
    "log_ret_1d", "log_ret_5d",
    # Momentum
    "roc_5", "roc_10", "roc_20",
    "momentum_5", "momentum_10",
    "price_accel",
    # Oscillators
    "rsi_7", "rsi_14", "rsi_21", "stoch_rsi",
    "stoch_k_14", "stoch_d_14",
    "williams_r", "cci",
    # MACD
    "macd_norm", "macd_hist",
    "ema_12_26_cross",
    # Bollinger
    "bb_pos_20", "bb_width_20", "bb_squeeze_20",
    "bb_pos_50", "bb_width_50",
    # Volatility
    "atr_pct_7", "atr_pct_14", "vol_10d", "vol_20d", "vol_ratio",
    "gk_vol",
    # Volume
    "vol_ratio_5_20", "vol_zscore", "vol_price_trend", "obv_trend",
    # Price action
    "body_size", "upper_shadow", "lower_shadow", "body_to_range",
    "gap_up", "gap_down", "doji", "inside_bar",
    # S/R
    "dist_support_20", "dist_resist_20", "sr_ratio_20",
    "dist_support_50", "dist_resist_50",
    # Trend regime
    "regime", "regime_bars",
    "above_cloud",
    # Efficiency
    "efficiency_ratio_10", "efficiency_ratio_20",
    "noise_20",
    # Z-score
    "zscore_20", "zscore_50",
    "price_rank_52w",
    # Calendar
    "dow", "month", "is_month_start", "is_month_end",
    "is_fiscal_q1", "is_fiscal_q4",
    # Sentiment
    "sentiment_score", "sentiment_x_momentum", "sentiment_x_rsi",
    # Lags (price level normalised via returns)
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
    "close_lag_1",  # absolute level
    # Skew/Kurt
    "ret_skew_20", "ret_kurt_20",
    # Crosses
    "sma_5_20_cross", "sma_10_50_cross",
]


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return feature columns that actually exist in df."""
    return [c for c in BASE_FEATURES if c in df.columns]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing OHLCV columns with close price as proxy."""
    cl = df["close"].astype(float)
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = cl
        else:
            df[col] = df[col].astype(float).fillna(cl)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    else:
        df["volume"] = df["volume"].astype(float).fillna(0.0)
    return df


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss.replace(0, np.nan))
    return 100 - 100 / (1 + rs)


def _atr(hi: pd.Series, lo: pd.Series, cl: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        hi - lo,
        (hi - cl.shift()).abs(),
        (lo - cl.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _streak(series: pd.Series) -> pd.Series:
    """Count consecutive identical values (regime persistence)."""
    out = []
    count = 0
    prev = None
    for v in series:
        if v == prev:
            count += 1
        else:
            count = 1
            prev = v
        out.append(count)
    return pd.Series(out, index=series.index, dtype=float)