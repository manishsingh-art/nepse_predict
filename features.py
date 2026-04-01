#!/usr/bin/env python3
"""
features.py — Comprehensive Feature Engineering for NEPSE
===========================================================
Produces 120+ features across:
  - Price action (OHLCV-derived)
  - Technical indicators (trend, momentum, volatility, volume)
  - Market microstructure (spreads, efficiency)
  - Nepal calendar effects (BS dates, NEPSE weekdays, fiscal year)
  - Festival & holiday proximity (Dashain, Tihar, Holi, etc.)
  - Nepal Bandh / market closure awareness
  - Cross-asset proxies (sentiment score injection)
  - Regime detection (hidden Markov-like heuristic)

All original features preserved. Nepal calendar features added on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, Any


# ─── Nepal Calendar Integration ───────────────────────────────────────────────
try:
    from nepal_calendar import NepalMarketCalendar, get_calendar
    _CAL: Optional[NepalMarketCalendar] = None

    def _get_cal() -> NepalMarketCalendar:
        global _CAL
        if _CAL is None:
            _CAL = NepalMarketCalendar(fetch_live=False)
        return _CAL

    _HAS_NEPAL_CAL = True
except ImportError:
    _HAS_NEPAL_CAL = False

    def _get_cal():
        return None


# ─── Main Feature Builder ─────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    sentiment_score: float = 0.0,
    smart_money_info: Optional[Dict[str, Any]] = None,
    include_context_features: bool = False,
) -> pd.DataFrame:
    """
    Adds all features to a copy of `df`.
    `sentiment_score` is a scalar [-1..+1] from news analysis.
    Returns DataFrame with all original columns + feature columns.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _ensure_ohlcv(df)

    cl  = df["close"].astype(float)
    op  = df["open"].astype(float)
    hi  = df["high"].astype(float)
    lo  = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # ── Returns ───────────────────────────────────────────────────────────────
    for p in [1, 2, 3, 5, 7, 10, 15, 20]:
        df[f"ret_{p}d"]     = cl.pct_change(p, fill_method=None)
        df[f"log_ret_{p}d"] = np.log(cl / cl.shift(p))

    # ── Moving Averages ───────────────────────────────────────────────────────
    for w in [5, 10, 20, 50, 100]:
        df[f"sma_{w}"]      = cl.rolling(w).mean()
        df[f"ema_{w}"]      = cl.ewm(span=w, adjust=False).mean()
        df[f"dist_sma_{w}"] = (cl - df[f"sma_{w}"]) / (df[f"sma_{w}"] + 1e-9)
        df[f"dist_ema_{w}"] = (cl - df[f"ema_{w}"]) / (df[f"ema_{w}"] + 1e-9)

    # MA crossover signals
    df["sma_5_20_cross"]  = (df["sma_5"]  > df["sma_20"]).astype(float)
    df["sma_10_50_cross"] = (df["sma_10"] > df["sma_50"]).astype(float)
    df["ema_12_26_cross"] = (
        cl.ewm(span=12, adjust=False).mean() > cl.ewm(span=26, adjust=False).mean()
    ).astype(float)

    # ── MACD ──────────────────────────────────────────────────────────────────
    ema12 = cl.ewm(span=12, adjust=False).mean()
    ema26 = cl.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_norm"]   = df["macd"] / (cl + 1e-9)

    # ── RSI family ────────────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"rsi_{period}"] = _rsi(cl, period)

    rsi14    = df["rsi_14"]
    rsi_min  = rsi14.rolling(14).min()
    rsi_max  = rsi14.rolling(14).max()
    df["stoch_rsi"] = (rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    for w in [20, 50]:
        mid = cl.rolling(w).mean()
        std = cl.rolling(w).std()
        df[f"bb_upper_{w}"]   = mid + 2 * std
        df[f"bb_lower_{w}"]   = mid - 2 * std
        df[f"bb_width_{w}"]   = (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"]) / (mid + 1e-9)
        df[f"bb_pos_{w}"]     = (cl - df[f"bb_lower_{w}"]) / (df[f"bb_upper_{w}"] - df[f"bb_lower_{w}"] + 1e-9)
        df[f"bb_squeeze_{w}"] = (
            df[f"bb_width_{w}"] < df[f"bb_width_{w}"].rolling(250).quantile(0.1)
        ).astype(float)

    # ── ATR & Volatility ──────────────────────────────────────────────────────
    for period in [7, 14, 21]:
        df[f"atr_{period}"]     = _atr(hi, lo, cl, period)
        df[f"atr_pct_{period}"] = df[f"atr_{period}"] / (cl + 1e-9)

    df["vol_10d"]   = df["ret_1d"].rolling(10).std()
    df["vol_20d"]   = df["ret_1d"].rolling(20).std()
    df["vol_ratio"] = df["vol_10d"] / (df["vol_20d"] + 1e-9)

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

    # ── CCI ───────────────────────────────────────────────────────────────────
    tp     = (hi + lo + cl) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_dev = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - tp_ma) / (0.015 * tp_dev + 1e-9)

    # ── OBV & Volume Indicators ───────────────────────────────────────────────
    obv_vals = [0.0]
    for i in range(1, len(df)):
        if cl.iloc[i] > cl.iloc[i - 1]:
            obv_vals.append(obv_vals[-1] + vol.iloc[i])
        elif cl.iloc[i] < cl.iloc[i - 1]:
            obv_vals.append(obv_vals[-1] - vol.iloc[i])
        else:
            obv_vals.append(obv_vals[-1])
    df["obv"]       = obv_vals
    df["obv_ema"]   = pd.Series(obv_vals, index=df.index).ewm(span=10, adjust=False).mean()
    df["obv_trend"] = (df["obv"] > df["obv_ema"]).astype(float)

    df["vol_sma_5"]       = vol.rolling(5).mean()
    df["vol_sma_20"]      = vol.rolling(20).mean()
    df["vol_ratio_5_20"]  = df["vol_sma_5"] / (df["vol_sma_20"] + 1e-9)
    df["vol_zscore"]      = (vol - df["vol_sma_20"]) / (vol.rolling(20).std() + 1e-9)
    df["vol_price_trend"] = df["ret_1d"] * df["vol_zscore"]

    # ── Ichimoku Cloud (simplified) ───────────────────────────────────────────
    high9  = hi.rolling(9).max()
    low9   = lo.rolling(9).min()
    high26 = hi.rolling(26).max()
    low26  = lo.rolling(26).min()
    df["ichi_conv"]   = (high9 + low9) / 2
    df["ichi_base"]   = (high26 + low26) / 2
    df["ichi_span_a"] = ((df["ichi_conv"] + df["ichi_base"]) / 2).shift(26)
    df["ichi_span_b"] = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    df["above_cloud"] = (cl > df["ichi_span_a"].fillna(0)).astype(float)

    # ── Price Action Features ─────────────────────────────────────────────────
    df["body_size"]     = (cl - op).abs() / (cl + 1e-9)
    df["upper_shadow"]  = (hi - pd.concat([cl, op], axis=1).max(axis=1)) / (cl + 1e-9)
    df["lower_shadow"]  = (pd.concat([cl, op], axis=1).min(axis=1) - lo) / (cl + 1e-9)
    df["body_to_range"] = (cl - op).abs() / (hi - lo + 1e-9)
    df["gap_up"]        = ((op > cl.shift(1)) & (op > cl.shift(1) * 1.005)).astype(float)
    df["gap_down"]      = ((op < cl.shift(1)) & (op < cl.shift(1) * 0.995)).astype(float)
    df["doji"]          = (df["body_size"] < 0.002).astype(float)
    df["inside_bar"]    = ((hi < hi.shift(1)) & (lo > lo.shift(1))).astype(float)

    # ── Support / Resistance Distance ────────────────────────────────────────
    for w in [10, 20, 50]:
        df[f"support_{w}"]      = lo.rolling(w).min()
        df[f"resist_{w}"]       = hi.rolling(w).max()
        df[f"dist_support_{w}"] = (cl - df[f"support_{w}"]) / (cl + 1e-9)
        df[f"dist_resist_{w}"]  = (df[f"resist_{w}"] - cl) / (cl + 1e-9)
        df[f"sr_ratio_{w}"]     = df[f"dist_support_{w}"] / (df[f"dist_resist_{w}"] + 1e-9)

    # ── Momentum & Rate of Change ─────────────────────────────────────────────
    for p in [5, 10, 20]:
        df[f"roc_{p}"]      = cl.pct_change(p, fill_method=None) * 100
        df[f"momentum_{p}"] = cl - cl.shift(p)

    df["price_accel"] = df["ret_1d"] - df["ret_1d"].shift(1)

    # ── Efficiency Ratio (Kaufman) ────────────────────────────────────────────
    for w in [10, 20]:
        direction       = (cl - cl.shift(w)).abs()
        volatility_sum  = cl.diff().abs().rolling(w).sum()
        df[f"efficiency_ratio_{w}"] = direction / (volatility_sum + 1e-9)

    # ── Noise / Fractal ───────────────────────────────────────────────────────
    for w in [20]:
        rets = df["ret_1d"].rolling(w)
        df[f"noise_{w}"] = rets.std() / (rets.apply(lambda x: np.mean(np.abs(x)), raw=True) + 1e-9)

    # ── Standard Calendar Features (AD) ──────────────────────────────────────
    df["dow"]            = df["date"].dt.dayofweek
    df["month"]          = df["date"].dt.month
    df["quarter"]        = df["date"].dt.quarter
    df["is_month_start"] = (df["date"].dt.day <= 5).astype(float)
    df["is_month_end"]   = (df["date"].dt.day >= 25).astype(float)
    df["is_fiscal_q1"]   = df["month"].isin([7, 8, 9]).astype(float)
    df["is_fiscal_q4"]   = df["month"].isin([4, 5, 6]).astype(float)
    for d_idx in range(5):
        df[f"dow_{d_idx}"] = (df["dow"] == d_idx).astype(float)

    # ── Nepal Calendar Features ───────────────────────────────────────────────
    if _HAS_NEPAL_CAL:
        df = _add_nepal_calendar_features(df)

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

    # ── Regime Detection (Advanced v5.0) ──────────────────────────────────────
    df["volatility_20"] = cl.pct_change().rolling(20).std()
    df["vol_ratio_20"]  = vol / (vol.rolling(20).mean() + 1e-9)
    # Simple Heuristic Regime
    df["is_bull"] = ((cl > df["sma_20"]) & (df["sma_20"] > df["sma_50"])).astype(float)
    df["is_bear"] = ((cl < df["sma_20"]) & (df["sma_20"] < df["sma_50"])).astype(float)
    df["regime_volatility"] = df["volatility_20"] * df["vol_ratio_20"]

    # Smart-money snapshots are inference-time context only. They must not be
    # broadcast across historical rows during training because that leaks a
    # single present-time state into the full sample.
    if include_context_features:
        sm = smart_money_info or {}
        df["sm_buy_hhi"] = float(sm.get("buy_hhi", 0.0) or 0.0)
        df["sm_sell_hhi"] = float(sm.get("sell_hhi", 0.0) or 0.0)
        df["sm_buy_concentration"] = float(sm.get("buy_concentration", 0.0) or sm.get("buy_concentration_top5", 0.0) or 0.0)
        df["sm_sell_concentration"] = float(sm.get("sell_concentration", 0.0) or sm.get("sell_concentration_top5", 0.0) or 0.0)
        df["sm_trap_score"] = float(sm.get("trap_score", 0.0) or 0.0)
        df["sm_wash_trading_alert"] = float(bool(sm.get("wash_trading_alert", False)))
        df["sm_smart_money_flow"] = float(sm.get("smart_money_flow", 0.0) or 0.0)
        df["sm_broker_concentration"] = float(sm.get("broker_concentration", 0.0) or 0.0)
        df["sm_accumulation_flag"] = float(sm.get("accumulation_flag", 0.0) or 0.0)

    # ── Price-Volume Divergence (v6.1) ───────────────────────────────────────
    # Divergence: Price Up + Vol Down => bearish divergence; Price Down + Vol Up => bullish divergence
    price_trend_5 = cl.pct_change(5, fill_method=None)
    vol_trend_5 = vol.pct_change(5, fill_method=None)
    df["pv_confirmation"] = ((price_trend_5 > 0) & (vol_trend_5 > 0) | (price_trend_5 < 0) & (vol_trend_5 < 0)).astype(float)
    df["pv_divergence"] = np.select(
        [
            (price_trend_5 > 0) & (vol_trend_5 < 0),
            (price_trend_5 < 0) & (vol_trend_5 > 0),
        ],
        [-1.0, 1.0],
        default=0.0,
    )
    df["pv_divergence_score"] = df["ret_1d"].rolling(5).mean() / (df["vol_ratio_5_20"] + 1e-9)
    df["smart_money_prox"] = (df["ret_1d"] > 0).astype(float) * (df["vol_ratio_5_20"] - 1.0)

    # ── Rolling Skewness & Kurtosis (Behavioral) ──────────────────────────────
    df["ret_skew_20"] = df["ret_1d"].rolling(20).skew()
    df["ret_kurt_20"] = df["ret_1d"].rolling(20).kurt()

    # ── Behavioral Indicators (v6.0) ──────────────────────────────────────────
    # FOMO: Price accelerating upwards while RSI is already high
    df["fomo_index"] = (df["price_accel"] > 0).astype(float) * (df["rsi_14"] / 100.0) * df["vol_ratio_5_20"]
    
    # Panic: Price falling sharply on high relative volume
    df["panic_index"] = (df["ret_1d"] < -0.02).astype(float) * df["vol_zscore"].clip(lower=0)
    
    # Volatility Clustering (GARCH-like approximation v6.1)
    # Captures the "fat tails" and clustering by looking at the variance of squared returns
    df["vol_of_vol"] = df["ret_1d"].rolling(10).std().rolling(10).std()
    df["garch_vol"] = np.sqrt(df["ret_1d"].pow(2).ewm(span=20, adjust=False).mean())
    df["garch_cluster"] = (df["garch_vol"] / (df["garch_vol"].rolling(60).mean() + 1e-9)).clip(0, 5)

    # ── Market-relative features vs NEPSE index (beta/corr) ───────────────────
    if "nepse_ret_1d" in df.columns:
        idx_ret = df["nepse_ret_1d"].astype(float)
        stk_ret = df["ret_1d"].astype(float)

        for w in [20, 60]:
            df[f"corr_nepse_{w}"] = stk_ret.rolling(w).corr(idx_ret)

        # Beta (cov/var) over 60d window
        w = 60
        cov = stk_ret.rolling(w).cov(idx_ret)
        var = idx_ret.rolling(w).var()
        df["beta_nepse_60"] = cov / (var + 1e-9)

        # Relative strength: stock return minus index return
        df["rel_ret_1d"] = stk_ret - idx_ret

    # ── Liquidity / microstructure robustness ─────────────────────────────────
    # Low-liquidity flag: current volume below 20th percentile of last 60 sessions
    try:
        vol_p20 = vol.rolling(60).quantile(0.20)
        df["illiquid_flag"] = (vol < vol_p20).astype(float)
    except Exception:
        df["illiquid_flag"] = 0.0

    # Volume anomaly magnitude (absolute z-score)
    df["vol_z_abs"] = df["vol_zscore"].abs()

    # ── Numeric regime label + streak (used by some model heads) ─────────────
    #  1 = bull, -1 = bear, 0 = sideways/neutral
    df["regime"] = np.select([df["is_bull"] > 0.5, df["is_bear"] > 0.5], [1.0, -1.0], default=0.0)
    df["regime_bars"] = _streak(df["regime"])

    # ── 52-week rank ──────────────────────────────────────────────────────────
    rolling_min_252 = cl.rolling(min(252, len(df))).min()
    rolling_max_252 = cl.rolling(min(252, len(df))).max()
    df["price_rank_52w"] = (cl - rolling_min_252) / (rolling_max_252 - rolling_min_252 + 1e-9)

    return df


# ─── Nepal Calendar Feature Injection ────────────────────────────────────────

def _add_nepal_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised Nepal calendar feature computation for all rows in df.
    Adds 29 Nepal-specific features per row using the NepalMarketCalendar engine.
    """
    cal = _get_cal()
    if cal is None:
        return df

    # Build feature matrix row-by-row (calendar lookups are fast, no network)
    feature_rows = []
    for d in df["date"]:
        try:
            ad_date = d.date() if hasattr(d, "date") else d
            feats = cal.get_nepali_features(ad_date)
        except Exception:
            # Fallback zeros if date out of BS table range
            feats = {k: 0.0 for k in [
                "bs_year", "bs_month", "bs_day", "bs_day_of_month_norm",
                "bs_month_name_idx", "fiscal_month", "fiscal_quarter",
                "is_fiscal_q1", "is_fiscal_q2", "is_fiscal_q3", "is_fiscal_q4",
                "is_fiscal_month_start", "is_fiscal_month_end",
                "nepse_day_of_week", "is_trading_day", "is_pre_holiday",
                "is_post_holiday", "days_to_next_holiday", "festival_proximity",
                "in_dashain_tihar", "in_new_year_window", "in_dividend_season",
                "in_agm_season", "nrb_policy_month", "ad_month", "ad_quarter",
                "ad_day_of_year_norm", "is_week_start", "is_week_end",
            ]}
        feature_rows.append(feats)

    cal_df = pd.DataFrame(feature_rows, index=df.index)

    # Rename with np_ prefix to avoid collision with existing features
    rename_map = {
        col: f"np_{col}"
        for col in cal_df.columns
        if col not in ("ad_month", "ad_quarter")  # these overlap, use np_ versions
    }
    cal_df = cal_df.rename(columns=rename_map)

    # Add columns that don't already exist
    for col in cal_df.columns:
        if col not in df.columns:
            df[col] = cal_df[col].values

    return df


# ── Target Engineering ────────────────────────────────────────────────────────

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds prediction targets:
    - target_ret_1d     : next session close-to-close return (primary target)
    - target_direction  : 1=up, 0=down next session (classification helper)
    - target_next_close : next session close (derived output helper)
    - target_ret_5d     : 5-day forward return (signal evaluation)
    """
    cl = df["close"].astype(float)
    df["target_date"] = df["date"].shift(-1)
    df["target_next_close"] = cl.shift(-1)
    df["target_ret_1d"]     = cl.pct_change(fill_method=None).shift(-1)
    df["target_ret_5d"]     = cl.pct_change(5, fill_method=None).shift(-5)
    df["target_direction"]  = (df["target_ret_1d"] > 0).astype(int)
    return df


# ── Feature Selection ─────────────────────────────────────────────────────────

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
    "regime", "regime_bars", "is_bull", "is_bear", "regime_volatility",
    "pv_divergence", "pv_confirmation", "smart_money_prox",
    "above_cloud",
    # Efficiency
    "efficiency_ratio_10", "efficiency_ratio_20",
    "noise_20",
    # Z-score
    "zscore_20", "zscore_50",
    "price_rank_52w",
    # Standard calendar
    "dow", "month", "is_month_start", "is_month_end",
    "is_fiscal_q1", "is_fiscal_q4",
    # Behavioral (v6.1)
    "fomo_index", "panic_index", "garch_vol", "garch_cluster", "pv_divergence_score", "vol_of_vol",
    # Lags
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
    "close_lag_1",
    # Skew/Kurt
    "ret_skew_20", "ret_kurt_20",
    # Crosses
    "sma_5_20_cross", "sma_10_50_cross",
    # Market (NEPSE Index)
    "nepse_ret_1d", "nepse_rsi_14", "nepse_dist_ema_20", "nepse_vol_ratio",
    # Market-relative
    "corr_nepse_20", "corr_nepse_60", "beta_nepse_60", "rel_ret_1d",
    # Liquidity filters
    "illiquid_flag", "vol_z_abs",
]

# Nepal-specific features (only added when nepal_calendar is available)
NEPAL_FEATURES = [
    "np_bs_month",
    # Removed: `np_bs_day_of_month_norm` (overfits / calendar leakage risk)
    "np_fiscal_month",
    "np_fiscal_quarter",
    "np_is_fiscal_q1",
    "np_is_fiscal_q2",
    "np_is_fiscal_q3",
    "np_is_fiscal_q4",
    "np_is_fiscal_month_start",
    "np_is_fiscal_month_end",
    "np_nepse_day_of_week",
    "np_is_pre_holiday",
    "np_is_post_holiday",
    "np_days_to_next_holiday",
    "np_festival_proximity",
    "np_in_dashain_tihar",
    "np_in_new_year_window",
    "np_in_dividend_season",
    "np_in_agm_season",
    "np_nrb_policy_month",
    "np_is_week_start",
    "np_is_week_end",
]


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all available feature columns (base + Nepal) that exist in df."""
    all_feats = BASE_FEATURES + NEPAL_FEATURES
    cols = [c for c in all_feats if c in df.columns]

    # Deterministic pruning: drop features with extreme missingness or zero variance.
    # This improves stability across symbols without changing the overall pipeline logic.
    pruned = []
    for c in cols:
        s = df[c]
        try:
            missing = float(s.isna().mean())
            if missing > 0.50:
                continue
            v = float(np.nanvar(s.astype(float).values))
            if not np.isfinite(v) or v < 1e-12:
                continue
        except Exception:
            continue
        pruned.append(c)
    return pruned


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
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


PRIMARY_TARGET_COL = "target_ret_1d"


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and clean raw OHLCV data before feature engineering.
    Keeps zero-volume rows but removes impossible candles and duplicate dates.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = _ensure_ohlcv(out)
    out = out.dropna(subset=["date", "open", "high", "low", "close"])

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)

    invalid = (
        out[price_cols].le(0).any(axis=1)
        | (out["high"] < out["low"])
        | (out["high"] < out[["open", "close"]].max(axis=1))
        | (out["low"] > out[["open", "close"]].min(axis=1))
        | (out["volume"] < 0)
    )
    out = out.loc[~invalid].copy()
    out = out.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    out["zero_volume_flag"] = (out["volume"] <= 0).astype(float)
    return out


def add_market_features(df: pd.DataFrame, nepse_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds NEPSE Index features to the input DataFrame.
    Features: index return, index RSI, index distance to EMA, and volume ratio.
    """
    df = df.copy()
    nepse = nepse_df.copy()
    
    # Ensure OHLCV
    nepse = _ensure_ohlcv(nepse)
    
    # Calculate index features
    nepse['nepse_ret_1d'] = nepse['close'].pct_change()
    nepse['nepse_rsi_14'] = _rsi(nepse['close'], 14)
    nepse_ema = nepse['close'].ewm(span=20, adjust=False).mean()
    nepse['nepse_dist_ema_20'] = (nepse['close'] - nepse_ema) / (nepse_ema + 1e-9)
    
    nepse_vol_sma = nepse['volume'].rolling(20).mean()
    nepse['nepse_vol_ratio'] = nepse['volume'] / (nepse_vol_sma + 1e-9)
    
    # Normalize dates for merging
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    nepse['date_only'] = pd.to_datetime(nepse['date']).dt.date
    
    # Merge only necessary columns
    cols_to_merge = ['date_only', 'nepse_ret_1d', 'nepse_rsi_14', 'nepse_dist_ema_20', 'nepse_vol_ratio']
    df = pd.merge(df, nepse[cols_to_merge], on='date_only', how='left')
    
    return df.drop(columns=['date_only'])


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
    out = []
    count = 0
    prev = None
    for v in series:
        count = count + 1 if v == prev else 1
        prev = v
        out.append(count)
    return pd.Series(out, index=series.index, dtype=float)