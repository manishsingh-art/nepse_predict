#!/usr/bin/env python3
"""
features.py — Production Feature Engineering for NEPSE
=======================================================
Produces 120+ features across six categories:
  - Price action & returns          (OHLCV-derived)
  - Technical indicators            (trend, momentum, volatility, volume)
  - Market microstructure           (spreads, efficiency ratios)
  - Nepal calendar effects          (BS dates, NEPSE weekdays, fiscal year)
  - Festival & holiday proximity    (Dashain, Tihar, Holi, etc.)
  - Behavioural signals             (FOMO, panic, smart-money proxies)

──────────────────────────────────────────────────────────────────
TIMING CONTRACT  (read before modifying any feature)
──────────────────────────────────────────────────────────────────
This is an END-OF-DAY model.  Row t represents a NEPSE trading
session that has FULLY CLOSED.  The target `target_ret_1d[t]` is
the NEXT session's close-to-close return:

    target_ret_1d[t] = close[t+1] / close[t] − 1

All features at row t MUST use ONLY information available after
session t has closed (close[t] and earlier).  Specifically:

  ✓  close[t], high[t], low[t], open[t], volume[t]
     — the session has ended; all observable.
  ✓  ret_1d[t] = close[t]/close[t-1] − 1
     — current session's return; fully observable at EOD.
  ✓  rolling(w).mean()[t]  over [close[t-w+1] … close[t]]
     — entirely backward-looking.
  ✗  Any reference to close[t+1] or later
     — strict look-ahead leakage.

Special cases:
  • Lag features (close_lag_k, ret_lag_k) use shift(k ≥ 1) — strictly past.
  • `illiquid_flag` uses a SHIFTED rolling quantile:
        vol.shift(1).rolling(60).quantile(0.20)
    so current volume is EXCLUDED from its own threshold (LEAK-02 fix).
  • `bb_squeeze` uses a SHIFTED quantile on BB-width for the same reason
    (LEAK-03 fix).
  • `garch_cluster` uses shift(1) on the normalisation denominator:
        garch_vol / garch_vol.shift(1).rolling(60).mean()
    so garch_vol[t] is NOT included in its own 60-bar normalisation
    window — consistent with the self-inclusion fixes above (LEAK-04).
  • RSI and ATR use Wilder's exponential smoothing: ewm(com=period-1)
    matching the industry standard (TradingView / Bloomberg). The old
    SMA-based RSI and span-based ATR are replaced (CORRECT-01/02 fix).

──────────────────────────────────────────────────────────────────
FRAGMENTATION CONTRACT  (FRAG-01 fix)
──────────────────────────────────────────────────────────────────
All engineered columns are accumulated in a plain Python dict and
concatenated onto the base DataFrame in a SINGLE pd.concat call at
the end of build_features.  This eliminates the 120+ individual
`df[col] = …` assignments that previously triggered pandas
PerformanceWarning about DataFrame fragmentation and caused
incremental memory reallocations on every assignment.

──────────────────────────────────────────────────────────────────
FEATURE SELECTION CONTRACT  (get_feature_cols)
──────────────────────────────────────────────────────────────────
`get_feature_cols(df, train_df=None)` prunes columns by variance and
missingness.  Pass `train_df` (the training split only) so that the
pruning statistics do not leak test-set distributional information
back into column selection (LEAK-01 fix).

──────────────────────────────────────────────────────────────────
PERFORMANCE NOTES
──────────────────────────────────────────────────────────────────
  • OBV:     vectorised via np.sign(diff) × volume → cumsum  (PERF-01)
  • _streak: vectorised via ne().cumsum() + groupby.cumcount  (PERF-02)
  • noise_w: vectorised via abs().rolling().mean()             (PERF-03)
  • Dict accumulator + single concat eliminates fragmentation   (FRAG-01)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ─── Nepal Calendar Integration ───────────────────────────────────────────────

try:
    from nepal_calendar import NepalMarketCalendar  # noqa: F401

    _CAL: Optional[NepalMarketCalendar] = None

    def _get_cal() -> Optional[NepalMarketCalendar]:
        global _CAL
        if _CAL is None:
            _CAL = NepalMarketCalendar(fetch_live=False)
        return _CAL

    _HAS_NEPAL_CAL = True

except ImportError:
    _HAS_NEPAL_CAL = False

    def _get_cal() -> None:  # type: ignore[misc]
        return None


# ─── Main Feature Builder ─────────────────────────────────────────────────────


def build_features(
    df: pd.DataFrame,
    sentiment_score: float = 0.0,
    smart_money_info: Optional[Dict[str, Any]] = None,
    include_context_features: bool = False,
) -> pd.DataFrame:
    """
    Engineer all features for a single NEPSE symbol.

    Parameters
    ----------
    df : pd.DataFrame
        Clean OHLCV frame with columns ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.  Sorting by date is enforced internally.
    sentiment_score : float
        Scalar news sentiment in [-1, +1].  Only used when
        ``include_context_features=True``.
    smart_money_info : dict, optional
        Floorsheet broker-analysis signals.  Attached as static context
        only when ``include_context_features=True`` to prevent broadcasting
        a single present-time snapshot across all historical training rows.
    include_context_features : bool
        When ``True``, attaches smart-money snapshot columns.  Must be
        ``False`` during model training (see TIMING CONTRACT above).

    Returns
    -------
    pd.DataFrame
        Original columns plus all engineered feature columns.
        Index is reset; frame is sorted ascending by ``date``.

    Notes
    -----
    All features at row *t* use only OHLCV data from session *t* and
    earlier.  See module docstring for the full timing contract.

    All engineered columns are accumulated in a dict and joined via a
    single ``pd.concat`` call, eliminating DataFrame fragmentation
    (FRAG-01).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _ensure_ohlcv(df)

    cl  = df["close"].astype(float)
    op  = df["open"].astype(float)
    hi  = df["high"].astype(float)
    lo  = df["low"].astype(float)
    vol = df["volume"].astype(float)

    # Accumulate ALL engineered columns here; pd.concat once at the end (FRAG-01).
    cols: Dict[str, pd.Series] = {}

    # ── 1. Returns ────────────────────────────────────────────────────────────
    # ret_pd[t] = close[t]/close[t-p] - 1.  Uses data through close[t] — OK
    # for an end-of-day model where session t has already closed.
    ret_1d = cl.pct_change(1, fill_method=None)
    cols["ret_1d"] = ret_1d
    for p in [2, 3, 5, 7, 10, 15, 20]:
        cols[f"ret_{p}d"] = cl.pct_change(p, fill_method=None)
    for p in [1, 2, 3, 5, 7, 10, 15, 20]:
        cols[f"log_ret_{p}d"] = np.log(cl / cl.shift(p))

    # ── 2. Moving Averages ────────────────────────────────────────────────────
    sma: Dict[int, pd.Series] = {}
    ema_w: Dict[int, pd.Series] = {}
    for w in [5, 10, 20, 50, 100]:
        s = cl.rolling(w).mean()
        e = cl.ewm(span=w, adjust=False).mean()
        sma[w]   = s
        ema_w[w] = e
        cols[f"sma_{w}"]      = s
        cols[f"ema_{w}"]      = e
        cols[f"dist_sma_{w}"] = (cl - s) / (s + 1e-9)
        cols[f"dist_ema_{w}"] = (cl - e) / (e + 1e-9)

    # MA crossover signals — reuse already-computed series
    cols["sma_5_20_cross"]  = (sma[5]  > sma[20]).astype(float)
    cols["sma_10_50_cross"] = (sma[10] > sma[50]).astype(float)
    ema12 = cl.ewm(span=12, adjust=False).mean()
    ema26 = cl.ewm(span=26, adjust=False).mean()
    cols["ema_12_26_cross"] = (ema12 > ema26).astype(float)

    # ── 3. MACD ───────────────────────────────────────────────────────────────
    macd     = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    cols["macd"]        = macd
    cols["macd_signal"] = macd_sig
    cols["macd_hist"]   = macd - macd_sig
    cols["macd_norm"]   = macd / (cl + 1e-9)

    # ── 4. RSI Family — Wilder's Exponential Smoothing ────────────────────────
    # FIX (CORRECT-01): original used rolling(period).mean() (SMA-RSI).
    # Wilder's RSI standard: ewm(com=period-1) ↔ alpha=1/period, matching
    # TradingView, Bloomberg, and Investopedia definitions.
    rsi_cache: Dict[int, pd.Series] = {}
    for period in [7, 14, 21]:
        r = _rsi(cl, period)
        rsi_cache[period] = r
        cols[f"rsi_{period}"] = r

    rsi14    = rsi_cache[14]
    rsi_min  = rsi14.rolling(14).min()
    rsi_max  = rsi14.rolling(14).max()
    cols["stoch_rsi"] = (rsi14 - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # ── 5. Bollinger Bands ────────────────────────────────────────────────────
    for w in [20, 50]:
        mid      = cl.rolling(w).mean()
        std      = cl.rolling(w).std()
        bb_upper = mid + 2 * std
        bb_lower = mid - 2 * std
        bb_width = (bb_upper - bb_lower) / (mid + 1e-9)

        # FIX (LEAK-03): shift the quantile reference window by 1 so the
        # current BB-width is NOT included in the percentile it is compared
        # against.  Avoids self-inclusion without adding look-ahead bias.
        hist_q = bb_width.shift(1).rolling(250, min_periods=50).quantile(0.1)

        cols[f"bb_upper_{w}"]   = bb_upper
        cols[f"bb_lower_{w}"]   = bb_lower
        cols[f"bb_width_{w}"]   = bb_width
        cols[f"bb_pos_{w}"]     = (cl - bb_lower) / (bb_upper - bb_lower + 1e-9)
        cols[f"bb_squeeze_{w}"] = (bb_width < hist_q).astype(float)

    # ── 6. ATR & Volatility — Wilder's Smoothing ─────────────────────────────
    # FIX (CORRECT-02): original used ewm(span=period) → alpha=2/(period+1).
    # Wilder's ATR standard: ewm(com=period-1) → alpha=1/period.
    for period in [7, 14, 21]:
        a = _atr(hi, lo, cl, period)
        cols[f"atr_{period}"]     = a
        cols[f"atr_pct_{period}"] = a / (cl + 1e-9)

    vol_10d = ret_1d.rolling(10).std()
    vol_20d = ret_1d.rolling(20).std()
    cols["vol_10d"]   = vol_10d
    cols["vol_20d"]   = vol_20d
    cols["vol_ratio"] = vol_10d / (vol_20d + 1e-9)

    # Garman-Klass volatility estimator (more efficient than close-to-close)
    log_hl = np.log(hi / lo.replace(0, np.nan))
    log_co = np.log(cl / op.replace(0, np.nan))
    cols["gk_vol"] = (
        np.sqrt(0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2)
        .rolling(20)
        .mean()
    )

    # ── 7. Stochastic Oscillator ──────────────────────────────────────────────
    for period in [9, 14]:
        lo_n    = lo.rolling(period).min()
        hi_n    = hi.rolling(period).max()
        stoch_k = 100 * (cl - lo_n) / (hi_n - lo_n + 1e-9)
        cols[f"stoch_k_{period}"] = stoch_k
        cols[f"stoch_d_{period}"] = stoch_k.rolling(3).mean()

    # ── 8. Williams %R ────────────────────────────────────────────────────────
    hi14 = hi.rolling(14).max()
    lo14 = lo.rolling(14).min()
    cols["williams_r"] = -100 * (hi14 - cl) / (hi14 - lo14 + 1e-9)

    # ── 9. CCI (Commodity Channel Index) ─────────────────────────────────────
    # Mean Absolute Deviation (MAD) has no vectorised pandas equivalent;
    # raw=True passes numpy arrays to the lambda for maximum speed.
    tp     = (hi + lo + cl) / 3
    tp_ma  = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    cols["cci"] = (tp - tp_ma) / (0.015 * tp_mad + 1e-9)

    # ── 10. OBV (On-Balance Volume) — fully vectorised ────────────────────────
    # FIX (PERF-01): np.sign(diff) = {+1, 0, -1} correctly handles the
    # equal-close case; fillna(0) gives OBV[0] = 0 matching loop semantics.
    price_dir = np.sign(cl.diff()).fillna(0)
    obv       = (price_dir * vol).cumsum()
    obv_ema   = obv.ewm(span=10, adjust=False).mean()
    cols["obv"]       = obv
    cols["obv_ema"]   = obv_ema
    cols["obv_trend"] = (obv > obv_ema).astype(float)

    # ── 11. Volume Indicators ─────────────────────────────────────────────────
    vol_sma_5  = vol.rolling(5).mean()
    vol_sma_20 = vol.rolling(20).mean()
    vol_std_20 = vol.rolling(20).std()
    vol_zscore  = (vol - vol_sma_20) / (vol_std_20 + 1e-9)
    vol_rat_520 = vol_sma_5 / (vol_sma_20 + 1e-9)

    cols["vol_sma_5"]       = vol_sma_5
    cols["vol_sma_20"]      = vol_sma_20
    cols["vol_ratio_5_20"]  = vol_rat_520
    cols["vol_zscore"]      = vol_zscore
    cols["vol_price_trend"] = ret_1d * vol_zscore

    # ── 12. Ichimoku Cloud (simplified) ──────────────────────────────────────
    # .shift(26) moves Senkou Span 26 bars BACK — no look-ahead: we observe
    # the cloud value that was computed 26 bars ago, now used as context.
    high9  = hi.rolling(9).max()
    low9   = lo.rolling(9).min()
    high26 = hi.rolling(26).max()
    low26  = lo.rolling(26).min()
    ichi_conv   = (high9  + low9)  / 2
    ichi_base   = (high26 + low26) / 2
    ichi_span_a = ((ichi_conv + ichi_base) / 2).shift(26)
    ichi_span_b = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)

    cols["ichi_conv"]   = ichi_conv
    cols["ichi_base"]   = ichi_base
    cols["ichi_span_a"] = ichi_span_a
    cols["ichi_span_b"] = ichi_span_b
    cols["above_cloud"] = (cl > ichi_span_a.fillna(0)).astype(float)

    # ── 13. Price Action Features ─────────────────────────────────────────────
    candle_body  = (cl - op).abs()
    candle_range = hi - lo
    candle_max   = pd.concat([cl, op], axis=1).max(axis=1)
    candle_min   = pd.concat([cl, op], axis=1).min(axis=1)
    body_size_s  = candle_body / (cl + 1e-9)
    prev_close   = cl.shift(1)

    # FIX (CORRECT-03): removed redundant first condition from gap_up/down.
    cols["body_size"]     = body_size_s
    cols["upper_shadow"]  = (hi - candle_max) / (cl + 1e-9)
    cols["lower_shadow"]  = (candle_min - lo) / (cl + 1e-9)
    cols["body_to_range"] = candle_body / (candle_range + 1e-9)
    cols["gap_up"]        = (op > prev_close * 1.005).astype(float)
    cols["gap_down"]      = (op < prev_close * 0.995).astype(float)
    cols["doji"]          = (body_size_s < 0.002).astype(float)
    cols["inside_bar"]    = ((hi < hi.shift(1)) & (lo > lo.shift(1))).astype(float)

    # ── 14. Support / Resistance Distance ────────────────────────────────────
    for w in [10, 20, 50]:
        support  = lo.rolling(w).min()
        resist   = hi.rolling(w).max()
        dist_sup = (cl - support) / (cl + 1e-9)
        dist_res = (resist - cl)  / (cl + 1e-9)
        cols[f"support_{w}"]      = support
        cols[f"resist_{w}"]       = resist
        cols[f"dist_support_{w}"] = dist_sup
        cols[f"dist_resist_{w}"]  = dist_res
        cols[f"sr_ratio_{w}"]     = dist_sup / (dist_res + 1e-9)

    # ── 15. Momentum & Rate of Change ────────────────────────────────────────
    for p in [5, 10, 20]:
        cols[f"roc_{p}"]      = cl.pct_change(p, fill_method=None) * 100
        cols[f"momentum_{p}"] = cl - cl.shift(p)

    price_accel = ret_1d - ret_1d.shift(1)
    cols["price_accel"] = price_accel

    # ── 16. Kaufman Efficiency Ratio ─────────────────────────────────────────
    for w in [10, 20]:
        direction   = (cl - cl.shift(w)).abs()
        path_length = cl.diff().abs().rolling(w).sum()
        cols[f"efficiency_ratio_{w}"] = direction / (path_length + 1e-9)

    # ── 17. Noise Ratio — fully vectorised ───────────────────────────────────
    # FIX (PERF-03): mean(|x|) = (ret_1d.abs()).rolling(w).mean() — vectorised.
    for w in [20]:
        ret_abs_mean = ret_1d.abs().rolling(w).mean()
        cols[f"noise_{w}"] = ret_1d.rolling(w).std() / (ret_abs_mean + 1e-9)

    # ── 18. Standard Calendar Features (AD) ──────────────────────────────────
    dow_ser   = df["date"].dt.dayofweek
    month_ser = df["date"].dt.month
    cols["dow"]            = dow_ser
    cols["month"]          = month_ser
    cols["quarter"]        = df["date"].dt.quarter
    cols["is_month_start"] = (df["date"].dt.day <= 5).astype(float)
    cols["is_month_end"]   = (df["date"].dt.day >= 25).astype(float)
    cols["is_fiscal_q1"]   = month_ser.isin([7, 8, 9]).astype(float)
    cols["is_fiscal_q4"]   = month_ser.isin([4, 5, 6]).astype(float)
    for d_idx in range(5):
        cols[f"dow_{d_idx}"] = (dow_ser == d_idx).astype(float)

    # ── 19. Lag Features ─────────────────────────────────────────────────────
    # Strictly backward-looking by construction: shift(k) with k ≥ 1.
    for lag in [1, 2, 3, 4, 5, 7, 10, 15]:
        cols[f"close_lag_{lag}"] = cl.shift(lag)
        cols[f"ret_lag_{lag}"]   = ret_1d.shift(lag)
        cols[f"vol_lag_{lag}"]   = vol.shift(lag)

    # ── 20. Z-Score Normalised Price ─────────────────────────────────────────
    for w in [20, 50]:
        mu = cl.rolling(w).mean()
        sd = cl.rolling(w).std()
        cols[f"zscore_{w}"] = (cl - mu) / (sd + 1e-9)

    # ── 21. Regime Detection ─────────────────────────────────────────────────
    sma20         = sma[20]
    sma50         = sma[50]
    volatility_20 = ret_1d.rolling(20).std()        # same computation as vol_20d
    vol_ratio_20  = vol / (vol.rolling(20).mean() + 1e-9)
    is_bull = ((cl > sma20) & (sma20 > sma50)).astype(float)
    is_bear = ((cl < sma20) & (sma20 < sma50)).astype(float)

    cols["volatility_20"]     = volatility_20
    cols["vol_ratio_20"]      = vol_ratio_20
    cols["is_bull"]           = is_bull
    cols["is_bear"]           = is_bear
    cols["regime_volatility"] = volatility_20 * vol_ratio_20

    # ── 22. Price-Volume Divergence ───────────────────────────────────────────
    price_trend_5 = cl.pct_change(5, fill_method=None)
    vol_trend_5   = vol.pct_change(5, fill_method=None)

    pv_div_arr = np.select(
        [
            (price_trend_5 > 0) & (vol_trend_5 < 0),
            (price_trend_5 < 0) & (vol_trend_5 > 0),
        ],
        [-1.0, 1.0],
        default=0.0,
    )
    cols["pv_confirmation"] = (
        ((price_trend_5 > 0) & (vol_trend_5 > 0))
        | ((price_trend_5 < 0) & (vol_trend_5 < 0))
    ).astype(float)
    cols["pv_divergence"]       = pd.Series(pv_div_arr, index=cl.index)
    cols["pv_divergence_score"] = ret_1d.rolling(5).mean() / (vol_rat_520 + 1e-9)
    cols["smart_money_prox"]    = (ret_1d > 0).astype(float) * (vol_rat_520 - 1.0)

    # ── 23. Rolling Skewness & Kurtosis ──────────────────────────────────────
    cols["ret_skew_20"] = ret_1d.rolling(20).skew()
    cols["ret_kurt_20"] = ret_1d.rolling(20).kurt()

    # ── 24. Behavioural Indicators ────────────────────────────────────────────
    cols["fomo_index"] = (
        (price_accel > 0).astype(float)
        * (rsi14 / 100.0)
        * vol_rat_520
    )
    cols["panic_index"] = (
        (ret_1d < -0.02).astype(float) * vol_zscore.clip(lower=0)
    )

    # GARCH-like volatility clustering (approximation via squared-return EWM).
    # EWM series: each value incorporates the current bar's squared return — OK
    # for EOD since ret_1d[t] is fully observable.
    vol_of_vol = ret_1d.rolling(10).std().rolling(10).std()
    garch_vol  = np.sqrt(ret_1d.pow(2).ewm(span=20, adjust=False).mean())

    # FIX (LEAK-04): shift(1) the normalisation denominator so garch_vol[t]
    # is NOT included in its own 60-bar scaling window — mirrors the
    # self-inclusion fixes applied to bb_squeeze (LEAK-03) and illiquid_flag
    # (LEAK-02).  min_periods=10 allows partial scaling during the warm-up.
    garch_cluster = (
        garch_vol / (garch_vol.shift(1).rolling(60, min_periods=10).mean() + 1e-9)
    ).clip(0, 5)

    cols["vol_of_vol"]    = vol_of_vol
    cols["garch_vol"]     = garch_vol
    cols["garch_cluster"] = garch_cluster

    # ── 25. Market-Relative Features vs NEPSE Index ───────────────────────────
    if "nepse_ret_1d" in df.columns:
        idx_ret = df["nepse_ret_1d"].astype(float)
        for w in [20, 60]:
            cols[f"corr_nepse_{w}"] = ret_1d.rolling(w).corr(idx_ret)
        cov = ret_1d.rolling(60).cov(idx_ret)
        var = idx_ret.rolling(60).var()
        cols["beta_nepse_60"] = cov / (var + 1e-9)
        cols["rel_ret_1d"]    = ret_1d - idx_ret

    # ── 26. Liquidity / Microstructure ───────────────────────────────────────
    # FIX (LEAK-02): vol.shift(1).rolling(60) excludes current-bar volume from
    # the quantile threshold — no self-inclusion.
    vol_p20 = vol.shift(1).rolling(60, min_periods=20).quantile(0.20)
    cols["illiquid_flag"] = (vol < vol_p20).astype(float)
    cols["vol_z_abs"]     = vol_zscore.abs()

    # ── 27. Numeric Regime + Streak ──────────────────────────────────────────
    regime_arr    = np.select([is_bull > 0.5, is_bear > 0.5], [1.0, -1.0], default=0.0)
    regime_series = pd.Series(regime_arr, index=cl.index)
    cols["regime"]      = regime_series
    cols["regime_bars"] = _streak(regime_series)

    # ── 28. 52-Week Price Rank ────────────────────────────────────────────────
    # min_periods is capped to the actual window so short test frames (<50 rows)
    # don't raise a ValueError ("min_periods must be <= window").
    window_252  = min(252, len(df))
    min_p_252   = min(50, window_252)
    rolling_min = cl.rolling(window_252, min_periods=min_p_252).min()
    rolling_max = cl.rolling(window_252, min_periods=min_p_252).max()
    cols["price_rank_52w"] = (cl - rolling_min) / (rolling_max - rolling_min + 1e-9)

    # ── Concatenate ALL engineered columns in one shot (FRAG-01) ──────────────
    # Building a DataFrame from a dict of same-indexed Series avoids the 120+
    # incremental `df[col] = …` assignments that cause pandas fragmentation.
    result = pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)

    # ── 29. Smart-Money Context (inference-time only) ─────────────────────────
    # Scalar broadcasts are applied AFTER the main concat.  They are kept
    # separate because a plain dict cannot mix Series and scalars in
    # pd.DataFrame().  Broadcasting a present-time snapshot across all
    # training rows would inject look-ahead — guarded by include_context_features.
    if include_context_features:
        sm = smart_money_info or {}
        result["sm_buy_hhi"]              = float(sm.get("buy_hhi", 0.0) or 0.0)
        result["sm_sell_hhi"]             = float(sm.get("sell_hhi", 0.0) or 0.0)
        result["sm_buy_concentration"]    = float(
            sm.get("buy_concentration", 0.0)
            or sm.get("buy_concentration_top5", 0.0)
            or 0.0
        )
        result["sm_sell_concentration"]   = float(
            sm.get("sell_concentration", 0.0)
            or sm.get("sell_concentration_top5", 0.0)
            or 0.0
        )
        result["sm_trap_score"]           = float(sm.get("trap_score", 0.0) or 0.0)
        result["sm_wash_trading_alert"]   = float(bool(sm.get("wash_trading_alert", False)))
        result["sm_smart_money_flow"]     = float(sm.get("smart_money_flow", 0.0) or 0.0)
        result["sm_broker_concentration"] = float(sm.get("broker_concentration", 0.0) or 0.0)
        result["sm_accumulation_flag"]    = float(sm.get("accumulation_flag", 0.0) or 0.0)

    # ── 30. Nepal Calendar Features ──────────────────────────────────────────
    if _HAS_NEPAL_CAL:
        result = _add_nepal_calendar_features(result)

    return result


# ─── Nepal Calendar Feature Injection ────────────────────────────────────────


def _add_nepal_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorised Nepal calendar feature injection.

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame that already contains a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Input frame with up to 29 Nepal-specific ``np_*`` columns added via
        a single ``pd.concat`` call (FRAG-01: no per-column assignment loop).

    Notes
    -----
    Calendar lookups are performed via ``NepalMarketCalendar.get_nepali_features``
    which is a pure in-memory computation (no network I/O).  Dates outside
    the BS conversion table fall back to zeros with no exception propagation.
    """
    cal = _get_cal()
    if cal is None:
        return df

    _FALLBACK_KEYS = [
        "bs_year", "bs_month", "bs_day", "bs_day_of_month_norm",
        "bs_month_name_idx", "fiscal_month", "fiscal_quarter",
        "is_fiscal_q1", "is_fiscal_q2", "is_fiscal_q3", "is_fiscal_q4",
        "is_fiscal_month_start", "is_fiscal_month_end",
        "nepse_day_of_week", "is_trading_day", "is_pre_holiday",
        "is_post_holiday", "days_to_next_holiday", "festival_proximity",
        "in_dashain_tihar", "in_new_year_window", "in_dividend_season",
        "in_agm_season", "nrb_policy_month", "ad_month", "ad_quarter",
        "ad_day_of_year_norm", "is_week_start", "is_week_end",
    ]
    _FALLBACK_ROW = {k: 0.0 for k in _FALLBACK_KEYS}

    feature_rows = []
    for d in df["date"]:
        try:
            ad_date = d.date() if hasattr(d, "date") else d
            feats   = cal.get_nepali_features(ad_date)
        except Exception:
            feats = _FALLBACK_ROW.copy()
        feature_rows.append(feats)

    cal_df = pd.DataFrame(feature_rows, index=df.index)

    # Prefix all columns with np_ except ad_month / ad_quarter which overlap
    rename_map = {
        col: f"np_{col}"
        for col in cal_df.columns
        if col not in ("ad_month", "ad_quarter")
    }
    cal_df = cal_df.rename(columns=rename_map)

    # FIX (FRAG-01): collect only genuinely new columns and concat in one call
    # instead of a loop of `df[col] = cal_df[col].values` assignments.
    new_cal_cols = {col: cal_df[col] for col in cal_df.columns if col not in df.columns}
    if new_cal_cols:
        df = pd.concat([df, pd.DataFrame(new_cal_cols, index=df.index)], axis=1)

    return df


# ── Target Engineering ────────────────────────────────────────────────────────


def add_triple_barrier_targets(
    df: pd.DataFrame,
    price_col: str = "close",
    atr_col: str = "atr_14",
    horizon: int = 5,
    multiplier: float = 1.5,
) -> pd.Series:
    """
    Triple Barrier labeling per López de Prado (AFML, Ch. 3).

    For each row t, defines:
      - Upper barrier : close[t] + multiplier × ATR[t]
      - Lower barrier : close[t] − multiplier × ATR[t]
      - Time barrier  : t + horizon (sessions)

    Scans forward through the intrabar high/low path (not just close) so
    that barrier touches within a session are correctly captured.

    Labels
    ------
    +1 : upper barrier hit before lower barrier and before time barrier
    -1 : lower barrier hit before upper barrier and before time barrier
     0 : time barrier expires without either price barrier being touched
    NaN: fewer than ``horizon`` future bars available (end of dataset)
         or ATR is NaN (model warm-up period)

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame that already contains OHLCV columns plus ATR
        (produced by ``build_features``).
    price_col : str
        Entry-price column (default ``"close"``).
    atr_col : str
        ATR column used to set barrier width (default ``"atr_14"``).
        If absent, falls back to 2 % of close.
    horizon : int
        Maximum number of forward sessions to scan (default 5).
    multiplier : float
        ATR multiplier for barrier width (default 1.5).

    Returns
    -------
    pd.Series
        Integer-valued labels {+1, 0, −1, NaN} named
        ``"triple_barrier_label"``.

    Notes
    -----
    Fully vectorised via numpy matrix construction — O(n × horizon) time
    and memory, no Python row loop.  For horizon = 5 and n = 1 000, the
    two working matrices consume < 80 KB.

    Timing contract: the label at row t uses only ``hi[t+1 … t+horizon]``
    and ``lo[t+1 … t+horizon]`` — genuine forward data, never feature
    input.  These columns must remain in target-only lists and are
    automatically excluded by ``get_feature_cols()``.
    """
    cl  = df[price_col].astype(float).values
    atr = (
        np.abs(df[atr_col].astype(float).values)
        if atr_col in df.columns
        else cl * 0.02
    )
    hi  = df["high"].astype(float).values if "high" in df.columns else cl
    lo  = df["low"].astype(float).values  if "low"  in df.columns else cl
    n   = len(cl)

    # ── Build forward-price matrices (n × horizon) ────────────────────────
    # future_hi[t, k-1] = hi[t + k]  for k in [1 … horizon]
    # Rows near the end that have no future data stay NaN.
    future_hi = np.full((n, horizon), np.nan)
    future_lo = np.full((n, horizon), np.nan)
    for k in range(1, horizon + 1):
        future_hi[: n - k, k - 1] = hi[k:]
        future_lo[: n - k, k - 1] = lo[k:]

    # ── Barriers (column broadcast) ───────────────────────────────────────
    upper = (cl + multiplier * atr).reshape(-1, 1)  # (n, 1)
    lower = (cl - multiplier * atr).reshape(-1, 1)

    # ── Boolean hit matrices (n × horizon) ───────────────────────────────
    upper_hits = future_hi >= upper   # True where intraday high crossed upper
    lower_hits = future_lo <= lower   # True where intraday low crossed lower

    # ── First crossing index (INF = never crossed within horizon) ─────────
    # np.argmax returns 0 when no True exists; guarded by the any() check.
    INF = horizon + 1
    first_upper = np.where(upper_hits.any(axis=1), np.argmax(upper_hits, axis=1), INF)
    first_lower = np.where(lower_hits.any(axis=1), np.argmax(lower_hits, axis=1), INF)

    # ── Label assignment ──────────────────────────────────────────────────
    # Tie (same bar hits both): label 0 — ambiguous; time barrier governs.
    label = np.where(
        (first_upper == INF) & (first_lower == INF),  # neither hit → time
        0,
        np.where(
            first_upper < first_lower, 1,             # upper first → long win
            np.where(first_lower < first_upper, -1, 0),  # lower first → loss
        ),
    ).astype(float)

    # ── Invalidate tail and warm-up rows ──────────────────────────────────
    if horizon > 0:
        label[n - horizon :] = np.nan  # last `horizon` rows lack full path
    label[np.isnan(atr)] = np.nan      # ATR not yet converged

    return pd.Series(label, index=df.index, name="triple_barrier_label")


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach forward-looking prediction targets to the feature frame.

    All target columns use ``shift(-k)`` to look FORWARD in time.  They
    must NEVER be included as input features to the model.

    Targets added
    -------------
    target_ret_1d          : next-session close-to-close return  (primary)
    target_direction       : 1 if next-session return > 0, else 0
    target_next_close      : next-session closing price
    target_ret_5d          : 5-session forward return
    target_date            : calendar date of the target session
    triple_barrier_label   : path-aware label {+1, 0, −1} over 5-session
                             horizon with 1.5×ATR barriers (meta-labeling
                             training signal only — not a model input)

    Parameters
    ----------
    df : pd.DataFrame
        Feature frame produced by ``build_features`` (must already contain
        ``atr_14`` and OHLCV columns).

    Returns
    -------
    pd.DataFrame
        Input frame with all target columns appended.

    Notes
    -----
    ``triple_barrier_label`` is used exclusively as the training signal for
    the ``MetaLabeler`` in ``meta_labeling.py``.  The primary regressor
    continues to train on ``target_ret_1d``.  Both targets coexist so
    callers can selectively use either.
    """
    cl = df["close"].astype(float)
    df["target_date"]            = df["date"].shift(-1)
    df["target_next_close"]      = cl.shift(-1)
    df["target_ret_1d"]          = cl.pct_change(fill_method=None).shift(-1)
    df["target_ret_5d"]          = cl.pct_change(5, fill_method=None).shift(-5)
    df["target_direction"]       = (df["target_ret_1d"] > 0).astype(int)
    df["triple_barrier_label"]   = add_triple_barrier_targets(df)
    return df


# ── Feature Lists ─────────────────────────────────────────────────────────────
BASE_FEATURES: List[str] = [
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
    # Support / Resistance
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
    # Behavioural (v6.1)
    "fomo_index", "panic_index", "garch_vol", "garch_cluster",
    "pv_divergence_score", "vol_of_vol",
    # Lags
    "ret_lag_1", "ret_lag_2", "ret_lag_3", "ret_lag_5",
    "close_lag_1",
    # Skew / Kurt
    "ret_skew_20", "ret_kurt_20",
    # Crossovers
    "sma_5_20_cross", "sma_10_50_cross",
    # Market (NEPSE Index)
    "nepse_ret_1d", "nepse_rsi_14", "nepse_dist_ema_20", "nepse_vol_ratio",
    # Market-relative
    "corr_nepse_20", "corr_nepse_60", "beta_nepse_60", "rel_ret_1d",
    # Liquidity
    "illiquid_flag", "vol_z_abs",
]

# Nepal-specific features (only added when nepal_calendar is importable)
NEPAL_FEATURES: List[str] = [
    "np_bs_month",
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


# ── Feature Selection ─────────────────────────────────────────────────────────


def get_feature_cols(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    Return usable feature columns, pruned by variance and missingness.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature frame.  Used to check which columns *exist* in the
        frame that will be passed to the model at inference time.
    train_df : pd.DataFrame, optional
        Training-split frame only.  When supplied, all pruning statistics
        (missingness rate, variance) are computed on this frame rather than
        on ``df``.  This prevents feature-selection decisions from leaking
        test-set distributional information back into column selection
        (FIX LEAK-01).  When ``None``, falls back to ``df`` for backward
        compatibility with call sites that have not been updated.

    Returns
    -------
    list of str
        Feature column names that:
        - exist in ``df``,
        - have ≤ 50 % missing values in the reference frame, and
        - have meaningful variance (nanvar > 1e-12) in the reference frame.

    Notes
    -----
    Pass ``train_df=feature_frame.iloc[train_indices]`` from the walk-forward
    loop to eliminate feature-selection leakage.  The returned list is
    deterministic for a given ``df`` / ``train_df`` pair.
    """
    all_feats = BASE_FEATURES + NEPAL_FEATURES
    # Only consider columns that actually exist in the full frame (df).
    candidate_cols = [c for c in all_feats if c in df.columns]

    # Statistics are computed on the reference frame (train only, or full).
    ref = train_df if (train_df is not None and not train_df.empty) else df

    pruned: List[str] = []
    for col in candidate_cols:
        if col not in ref.columns:
            continue
        s = ref[col]
        try:
            if float(s.isna().mean()) > 0.50:
                continue
            v = float(np.nanvar(s.astype(float).values))
            if not np.isfinite(v) or v < 1e-12:
                continue
        except Exception:
            continue
        pruned.append(col)

    return pruned


# ── Data Cleaning ─────────────────────────────────────────────────────────────
PRIMARY_TARGET_COL = "target_ret_1d"


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee that open, high, low, volume columns exist and are numeric.

    Missing price columns are filled with ``close``; missing volume is
    filled with 0.  Does not modify the input frame (caller provides a copy).
    """
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


def clean_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise and validate raw OHLCV data before feature engineering.

    Removes rows with:
    - non-positive prices,
    - high < low or high < open/close or low > open/close (impossible candles),
    - negative volume.

    Duplicate dates are resolved by keeping the last occurrence.
    A ``zero_volume_flag`` column is added (1 when volume ≤ 0).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV frame as fetched from any data source.

    Returns
    -------
    pd.DataFrame
        Clean, sorted, deduplicated OHLCV frame with reset index.
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
        | (out["low"]  > out[["open", "close"]].min(axis=1))
        | (out["volume"] < 0)
    )
    out = out.loc[~invalid].copy()
    out = (
        out.sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    out["zero_volume_flag"] = (out["volume"] <= 0).astype(float)
    return out


def add_market_features(
    df: pd.DataFrame,
    nepse_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge NEPSE Index features into the stock's OHLCV frame.

    Features computed on the index and merged by date:
    - ``nepse_ret_1d``      : index daily return
    - ``nepse_rsi_14``      : index RSI-14 (Wilder's)
    - ``nepse_dist_ema_20`` : index distance from its 20-day EMA
    - ``nepse_vol_ratio``   : index volume / 20-day average volume

    Parameters
    ----------
    df : pd.DataFrame
        Stock OHLCV frame (clean).
    nepse_df : pd.DataFrame
        NEPSE Index OHLCV frame (clean), same date range or wider.

    Returns
    -------
    pd.DataFrame
        Stock frame with the four index feature columns left-joined by date.

    Notes
    -----
    All index features use data through the same session *t* as the stock
    frame — both frames represent fully closed sessions at merge time.
    """
    df    = df.copy()
    nepse = _ensure_ohlcv(nepse_df.copy())

    nepse_cl  = nepse["close"].astype(float)
    nepse_ema = nepse_cl.ewm(span=20, adjust=False).mean()

    nepse["nepse_ret_1d"]      = nepse_cl.pct_change()
    nepse["nepse_rsi_14"]      = _rsi(nepse_cl, 14)
    nepse["nepse_dist_ema_20"] = (nepse_cl - nepse_ema) / (nepse_ema + 1e-9)
    nepse["nepse_vol_ratio"]   = (
        nepse["volume"].astype(float)
        / (nepse["volume"].astype(float).rolling(20).mean() + 1e-9)
    )

    df["_date_key"]    = pd.to_datetime(df["date"]).dt.normalize()
    nepse["_date_key"] = pd.to_datetime(nepse["date"]).dt.normalize()

    merge_cols = [
        "_date_key",
        "nepse_ret_1d", "nepse_rsi_14",
        "nepse_dist_ema_20", "nepse_vol_ratio",
    ]
    df = pd.merge(df, nepse[merge_cols], on="_date_key", how="left")
    return df.drop(columns=["_date_key"])


# ── Technical Indicator Helpers ───────────────────────────────────────────────


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Wilder's Relative Strength Index.

    Uses exponential smoothing with alpha = 1/period (``ewm(com=period-1)``),
    matching the TradingView / Bloomberg standard.  The previous SMA-based
    implementation produced non-standard values that diverged from industry
    benchmarks and introduced abrupt boundary effects when extreme values
    rolled out of the window (FIX CORRECT-01).

    Parameters
    ----------
    series : pd.Series
        Price series (typically ``close``).
    period : int
        Smoothing period (default 14 as per Wilder's original paper).

    Returns
    -------
    pd.Series
        RSI values in [0, 100].  First ``period`` rows are NaN.
    """
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(
    hi: pd.Series,
    lo: pd.Series,
    cl: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Compute Wilder's Average True Range.

    Uses exponential smoothing with alpha = 1/period (``ewm(com=period-1)``),
    matching the Wilder (1978) standard.  The previous ``ewm(span=period)``
    used alpha = 2/(period+1), which overstates short-term noise and does
    not match most platform implementations (FIX CORRECT-02).

    Parameters
    ----------
    hi, lo, cl : pd.Series
        High, low, and close price series (same length, same index).
    period : int
        Smoothing period (default 14 as per Wilder's original paper).

    Returns
    -------
    pd.Series
        ATR values ≥ 0.  First ``period`` rows are NaN.
    """
    prev_cl = cl.shift(1)
    true_range = pd.concat(
        [hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(com=period - 1, min_periods=period).mean()


def _streak(series: pd.Series) -> pd.Series:
    """
    Count consecutive occurrences of the same value (1-indexed run length).

    Fully vectorised via ``ne().cumsum()`` + ``groupby.cumcount``.
    Replaces the original Python ``for`` loop which was O(n) interpreted
    and ~50× slower for long series (FIX PERF-02).

    Examples
    --------
    >>> _streak(pd.Series([1, 1, 1, -1, -1, 0, 1, 1])).tolist()
    [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0]

    Parameters
    ----------
    series : pd.Series
        Typically the ``regime`` column with values in {-1, 0, 1}.

    Returns
    -------
    pd.Series
        Float series of same length and index as input.
    """
    # fillna(True) ensures the very first row always starts a new group
    # (because shift(1) makes it NaN and NaN comparison is ambiguous).
    is_new_group = series.ne(series.shift(1)).fillna(True)
    group_id     = is_new_group.cumsum()
    return group_id.groupby(group_id).cumcount().add(1).astype(float)
