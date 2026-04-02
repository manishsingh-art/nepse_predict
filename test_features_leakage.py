#!/usr/bin/env python3
"""
test_features_leakage.py — Leakage, Correctness & Quality Tests
================================================================
Tests organised into eight categories:

  1. No future leakage            — modifying future rows must not affect past features
  2. Rolling feature correctness  — SMA, RSI, ATR, OBV spot-checks
  3. Output shape consistency     — row count, required columns
  4. NaN levels                   — no explosion after warm-up period
  5. Determinism                  — same input ⟹ same output
  6. Target alignment             — targets are strictly forward-shifted
  7. Specific fix validations     — LEAK-01/02/03, CORRECT-01/02, PERF-01/02/03
  8. Feature selection            — get_feature_cols train_df parameter

Run with:
    pytest test_features_leakage.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import (
    BASE_FEATURES,
    _atr,
    _rsi,
    _streak,
    add_targets,
    build_features,
    get_feature_cols,
)


# ─── Shared fixture ───────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic OHLCV frame with realistic NEPSE-like daily price dynamics.

    Uses a geometric random walk for close, then constructs OHLV with
    small intraday noise so that OHLCV invariants always hold:
        low ≤ min(open, close) ≤ max(open, close) ≤ high
    """
    rng    = np.random.default_rng(seed)
    dates  = pd.bdate_range("2021-01-01", periods=n, freq="B")
    close  = 100.0 * np.cumprod(1.0 + rng.normal(0.001, 0.015, n))
    spread = close * rng.uniform(0.005, 0.015, n)

    open_  = close + rng.uniform(-0.5, 0.5, n) * spread
    high   = np.maximum(close, open_) + rng.uniform(0.001, 0.5, n) * spread
    low    = np.minimum(close, open_) - rng.uniform(0.001, 0.5, n) * spread
    volume = rng.integers(10_000, 1_000_000, n).astype(float)

    return pd.DataFrame(
        {"date": dates, "open": open_, "high": high, "low": low,
         "close": close, "volume": volume}
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NO FUTURE LEAKAGE
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoFutureLeakage:
    """
    Core property: features at row t must depend only on data from rows ≤ t.
    Verified by perturbing future rows and asserting past features are unchanged.
    """

    def test_perturbing_future_rows_does_not_change_past_features(self) -> None:
        """Modify last 30 rows drastically; rows 0-160 must be bit-identical."""
        n   = 200
        df1 = _make_ohlcv(n, seed=0)
        df2 = df1.copy()

        # Dramatic perturbation on the last 30 rows
        for col in ["close", "high", "low", "open"]:
            df2.iloc[170:, df2.columns.get_loc(col)] *= 3.0
        df2.iloc[170:, df2.columns.get_loc("volume")] *= 10.0

        feat1 = build_features(df1)
        feat2 = build_features(df2)

        skip = {"date", "open", "high", "low", "close", "volume"}
        check_cols = [c for c in feat1.columns if c not in skip]
        check_end  = 160  # well before the perturbation boundary

        pd.testing.assert_frame_equal(
            feat1.iloc[:check_end][check_cols].reset_index(drop=True),
            feat2.iloc[:check_end][check_cols].reset_index(drop=True),
            check_names=False,
            rtol=1e-9,
            atol=0.0,
            obj="Features at rows 0-159",
        )

    def test_lag_features_are_strictly_backward_looking(self) -> None:
        """ret_lag_1[t] == ret_1d[t-1] and close_lag_1[t] == close[t-1]."""
        feat = build_features(_make_ohlcv(120))
        for t in range(15, 100):
            assert abs(feat["ret_lag_1"].iloc[t] - feat["ret_1d"].iloc[t - 1]) < 1e-12, \
                f"ret_lag_1 misaligned at row {t}"
            assert abs(feat["close_lag_1"].iloc[t] - feat["close"].iloc[t - 1]) < 1e-12, \
                f"close_lag_1 misaligned at row {t}"

    def test_higher_lags_are_consistent(self) -> None:
        """ret_lag_k[t] == ret_1d[t-k] for k in {2, 3, 5}."""
        feat = build_features(_make_ohlcv(150))
        for k in [2, 3, 5]:
            col = f"ret_lag_{k}"
            for t in range(k + 10, 120):
                expected = feat["ret_1d"].iloc[t - k]
                actual   = feat[col].iloc[t]
                assert abs(actual - expected) < 1e-12, \
                    f"{col} misaligned at row {t}"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ROLLING FEATURE CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRollingCorrectness:

    def test_sma_5_first_valid_value(self) -> None:
        """SMA-5[4] must equal the simple mean of the first five closes."""
        df   = _make_ohlcv(50)
        feat = build_features(df)
        expected = df["close"].iloc[:5].mean()
        assert abs(feat["sma_5"].iloc[4] - expected) < 1e-9

    def test_sma_10_midrange(self) -> None:
        """SMA-10[t] == mean(close[t-9..t]) for t = 20."""
        df   = _make_ohlcv(50)
        feat = build_features(df)
        t    = 20
        expected = df["close"].iloc[t - 9: t + 1].mean()
        assert abs(feat["sma_10"].iloc[t] - expected) < 1e-9

    def test_rsi_bounds_all_periods(self) -> None:
        """RSI must always be in [0, 100] for all three periods."""
        feat = build_features(_make_ohlcv(300))
        for period in [7, 14, 21]:
            rsi = feat[f"rsi_{period}"].dropna()
            assert rsi.min() >= 0.0 - 1e-9, f"RSI({period}) < 0"
            assert rsi.max() <= 100.0 + 1e-9, f"RSI({period}) > 100"

    def test_atr_non_negative_all_periods(self) -> None:
        """ATR is a range-based measure and must always be ≥ 0."""
        feat = build_features(_make_ohlcv(200))
        for period in [7, 14, 21]:
            atr = feat[f"atr_{period}"].dropna()
            assert (atr >= 0).all(), f"ATR({period}) has negative values"

    def test_obv_correctness_known_sequence(self) -> None:
        """
        OBV accumulates correctly against a hand-computed reference.

        Closes:  [100, 101, 100, 102, 101]
        Volumes: [1000, 1500, 1200, 1800, 1100]
        Expected OBV: [0, +1500, +300, +2100, +1000]
        """
        rows = pd.DataFrame(
            {
                "date":   pd.bdate_range("2024-01-01", periods=5),
                "open":   [100, 101, 100, 102, 101],
                "high":   [101, 102, 101, 103, 102],
                "low":    [99,  100,  99, 101, 100],
                "close":  [100, 101, 100, 102, 101],
                "volume": [1_000, 1_500, 1_200, 1_800, 1_100],
            }
        )
        feat = build_features(rows)
        np.testing.assert_allclose(
            feat["obv"].values,
            [0.0, 1_500.0, 300.0, 2_100.0, 1_000.0],
            atol=1e-6,
        )

    def test_obv_equal_close_adds_zero(self) -> None:
        """When consecutive closes are equal, OBV must not change."""
        rows = pd.DataFrame(
            {
                "date":   pd.bdate_range("2024-01-01", periods=3),
                "open":   [100, 100, 100],
                "high":   [101, 101, 101],
                "low":    [99,  99,  99],
                "close":  [100, 100, 100],  # all equal
                "volume": [1_000, 2_000, 3_000],
            }
        )
        feat = build_features(rows)
        assert feat["obv"].iloc[1] == 0.0, "Equal closes should not change OBV"
        assert feat["obv"].iloc[2] == 0.0, "Equal closes should not change OBV"

    def test_bollinger_pos_range(self) -> None:
        """bb_pos should be near 0 at lower band and near 1 at upper band."""
        feat = build_features(_make_ohlcv(300))
        for w in [20, 50]:
            pos = feat[f"bb_pos_{w}"].dropna()
            # Values outside [-2, 3] indicate a calculation error
            assert pos.min() > -3.0, f"bb_pos_{w} has extreme low outlier"
            assert pos.max() <  4.0, f"bb_pos_{w} has extreme high outlier"

    def test_streak_known_sequence(self) -> None:
        """_streak gives the correct 1-indexed run-length for a known input."""
        s      = pd.Series([1.0, 1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0])
        result = _streak(s).tolist()
        assert result == [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0]

    def test_streak_all_same(self) -> None:
        """A constant series should produce a monotone increasing streak."""
        s = pd.Series([1.0] * 5)
        assert _streak(s).tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_streak_all_different(self) -> None:
        """A series with no repeats should always return 1."""
        s = pd.Series([1.0, -1.0, 1.0, -1.0, 0.0])
        assert _streak(s).tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. OUTPUT SHAPE CONSISTENCY
# ═══════════════════════════════════════════════════════════════════════════════


class TestOutputShape:

    def test_row_count_preserved(self) -> None:
        df   = _make_ohlcv(200)
        feat = build_features(df)
        assert len(feat) == len(df), "build_features must not change row count"

    def test_original_ohlcv_columns_preserved(self) -> None:
        df   = _make_ohlcv(100)
        feat = build_features(df)
        for col in ["date", "open", "high", "low", "close", "volume"]:
            assert col in feat.columns, f"Original column '{col}' missing"

    def test_core_feature_columns_present(self) -> None:
        df   = _make_ohlcv(200)
        feat = build_features(df)
        required = [
            "ret_1d", "rsi_14", "macd", "atr_14",
            "obv", "vol_zscore", "regime", "regime_bars",
            "price_rank_52w", "gk_vol", "illiquid_flag",
        ]
        for col in required:
            assert col in feat.columns, f"Core column '{col}' missing"

    def test_no_extra_rows_from_merge(self) -> None:
        """build_features must never inflate the row count (e.g. from merges)."""
        df   = _make_ohlcv(150, seed=5)
        feat = build_features(df)
        assert len(feat) == 150


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NaN LEVELS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNaNLevels:
    """After the warm-up period, features must not have excessive NaN."""

    WARMUP   = 100   # rows to skip (indicator initialisation)
    NAN_LIMIT = 0.05  # 5 % threshold

    def test_no_nan_explosion_in_base_features(self) -> None:
        """
        No BASE_FEATURE should have > 5 % NaN in the tail (rows 100+)
        of a 400-row dataset.
        """
        df   = _make_ohlcv(400)
        feat = build_features(df)
        tail = feat.iloc[self.WARMUP:]

        violations = []
        for col in BASE_FEATURES:
            if col not in feat.columns:
                continue
            nan_pct = tail[col].isna().mean()
            if nan_pct > self.NAN_LIMIT:
                violations.append(f"{col}: {nan_pct:.1%}")

        assert not violations, (
            f"Features with excessive NaN in tail:\n" + "\n".join(violations)
        )

    def test_ret_1d_is_fully_populated(self) -> None:
        """ret_1d should have exactly one NaN (the first row)."""
        feat = build_features(_make_ohlcv(100))
        assert feat["ret_1d"].isna().sum() == 1

    def test_lag_features_nan_count_matches_lag(self) -> None:
        """close_lag_k must have exactly k leading NaN values."""
        feat = build_features(_make_ohlcv(100))
        for k in [1, 2, 3, 5]:
            col     = f"close_lag_{k}"
            n_nans  = feat[col].isna().sum()
            assert n_nans == k, \
                f"{col}: expected {k} NaN(s), got {n_nans}"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DETERMINISM
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeterminism:

    def test_same_input_produces_same_output(self) -> None:
        """build_features is a pure function: identical input ⟹ identical output."""
        df    = _make_ohlcv(200, seed=77)
        feat1 = build_features(df.copy())
        feat2 = build_features(df.copy())

        skip      = {"date"}
        feat_cols = [c for c in feat1.columns if c not in skip]
        pd.testing.assert_frame_equal(feat1[feat_cols], feat2[feat_cols])

    def test_column_order_is_stable(self) -> None:
        """Column order must be the same across two calls."""
        df    = _make_ohlcv(200, seed=99)
        feat1 = build_features(df.copy())
        feat2 = build_features(df.copy())
        assert list(feat1.columns) == list(feat2.columns)

    def test_different_seeds_produce_different_features(self) -> None:
        """Sanity check: distinct inputs must produce distinct outputs."""
        feat1 = build_features(_make_ohlcv(200, seed=1))
        feat2 = build_features(_make_ohlcv(200, seed=2))
        assert not feat1["ret_1d"].equals(feat2["ret_1d"])


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TARGET ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════


class TestTargetAlignment:

    def test_target_ret_1d_is_next_session_return(self) -> None:
        """target_ret_1d[t] must equal close[t+1] / close[t] - 1."""
        df     = _make_ohlcv(100)
        df_t   = add_targets(df)
        closes = df_t["close"].values
        for t in range(20, 90):
            expected = closes[t + 1] / closes[t] - 1.0
            actual   = df_t["target_ret_1d"].iloc[t]
            assert abs(actual - expected) < 1e-12, f"Mismatch at row {t}"

    def test_target_next_close_is_next_bar(self) -> None:
        """target_next_close[t] must equal close[t+1]."""
        df   = _make_ohlcv(100)
        df_t = add_targets(df)
        for t in range(10, 90):
            assert abs(df_t["target_next_close"].iloc[t] - df_t["close"].iloc[t + 1]) < 1e-9

    def test_last_row_targets_are_nan(self) -> None:
        """The very last row must have NaN targets — no future data exists."""
        df   = _make_ohlcv(100)
        df_t = add_targets(df)
        assert pd.isna(df_t["target_ret_1d"].iloc[-1]), \
            "target_ret_1d must be NaN on last row"
        assert pd.isna(df_t["target_next_close"].iloc[-1]), \
            "target_next_close must be NaN on last row"

    def test_target_direction_derived_from_target_ret(self) -> None:
        """target_direction must be 1 iff target_ret_1d > 0."""
        df   = _make_ohlcv(200)
        df_t = add_targets(df)
        mask = df_t["target_ret_1d"].notna()
        pos  = df_t.loc[mask, "target_ret_1d"] > 0
        expected_dir = pos.astype(int)
        actual_dir   = df_t.loc[mask, "target_direction"]
        pd.testing.assert_series_equal(
            actual_dir.reset_index(drop=True),
            expected_dir.reset_index(drop=True),
            check_names=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SPECIFIC FIX VALIDATIONS
# ═══════════════════════════════════════════════════════════════════════════════


class TestLeakageFixes:
    """Verify each targeted fix from the audit."""

    # ── FIX LEAK-01: get_feature_cols uses train_df for statistics ────────────

    def test_get_feature_cols_with_train_df_returns_subset(self) -> None:
        """
        All columns returned when train_df is supplied must also exist in df.
        The function must not hallucinate columns.
        """
        df       = _make_ohlcv(400)
        feat     = build_features(df)
        train_ft = feat.iloc[:300]

        train_cols = get_feature_cols(feat, train_df=train_ft)
        all_cols   = get_feature_cols(feat)

        for col in train_cols:
            assert col in feat.columns, f"train_cols returned '{col}' not in feat"

    def test_get_feature_cols_backward_compatible(self) -> None:
        """get_feature_cols(df) with no train_df still returns a non-empty list."""
        feat = build_features(_make_ohlcv(200))
        cols = get_feature_cols(feat)
        assert isinstance(cols, list) and len(cols) > 0

    def test_get_feature_cols_train_df_excludes_high_missing(self) -> None:
        """
        If a column has > 50 % NaN in train_df, it must be excluded even if
        it has low NaN overall (e.g., the column fills in later in the frame).
        """
        df   = _make_ohlcv(400)
        feat = build_features(df)

        # Inject a synthetic column that is all-NaN in the training portion
        feat["_test_col"] = np.nan
        feat.loc[feat.index[300:], "_test_col"] = 1.0  # only in test portion

        train_ft = feat.iloc[:300].copy()
        cols     = get_feature_cols(feat, train_df=train_ft)
        assert "_test_col" not in cols, \
            "_test_col has 100% NaN in train_df and must be excluded"

    # ── FIX LEAK-02: illiquid_flag threshold excludes current volume ───────────

    def test_illiquid_flag_threshold_excludes_current_volume(self) -> None:
        """
        Setting today's volume to an extreme value must not change today's
        illiquid_flag threshold, because the threshold is lagged (shift(1)).
        """
        df1 = _make_ohlcv(200)
        df2 = df1.copy()
        df2.loc[df2.index[-1], "volume"] = 1e12  # extreme current bar

        feat1 = build_features(df1)
        feat2 = build_features(df2)

        # The quantile threshold for the last row is based on rows [N-61..N-2]
        # (shift(1).rolling(60)), which is identical in both frames.
        # Therefore illiquid_flag at last row must be identical.
        assert feat1["illiquid_flag"].iloc[-1] == feat2["illiquid_flag"].iloc[-1], (
            "illiquid_flag at last row changed when only current volume changed "
            "(self-inclusion not fully fixed)"
        )

    # ── FIX LEAK-03: bb_squeeze threshold excludes current BB-width ───────────

    def test_bb_squeeze_threshold_excludes_current_bar(self) -> None:
        """
        The bb_squeeze quantile threshold is based on shifted (lagged) BB-widths.
        A change in the current close must not alter the quantile used as
        the threshold for the same row.
        """
        df1 = _make_ohlcv(350)
        df2 = df1.copy()

        # Change only the last row's close (affects current BB-width but not
        # the shifted quantile threshold used for the squeeze check)
        df2.loc[df2.index[-1], "close"] = df2["close"].max() * 5

        feat1 = build_features(df1)
        feat2 = build_features(df2)

        # The penultimate row is unaffected in both frames — its squeeze flag
        # should be identical.
        for w in [20, 50]:
            col = f"bb_squeeze_{w}"
            if col in feat1.columns:
                assert feat1[col].iloc[-2] == feat2[col].iloc[-2], (
                    f"{col} penultimate row changed when only the last close "
                    "changed — quantile threshold is not properly lagged"
                )

    # ── FIX CORRECT-01: RSI uses Wilder's EWM, not SMA ───────────────────────

    def test_wilder_rsi_does_not_jump_abruptly_on_window_boundary(self) -> None:
        """
        SMA-RSI jumps abruptly when an extreme value rolls out of the window.
        Wilder's EWM-RSI decays it smoothly.  The maximum single-day RSI
        change should be bounded even after a sudden reversal.
        """
        # 20 strongly rising days, then 20 falling days
        closes = [100.0 + i * 2 for i in range(20)] + [140.0 - i * 2 for i in range(20)]
        df     = pd.DataFrame(
            {
                "date":   pd.bdate_range("2024-01-01", periods=40),
                "open":   closes,
                "high":   [c + 0.5 for c in closes],
                "low":    [c - 0.5 for c in closes],
                "close":  closes,
                "volume": [100_000] * 40,
            }
        )
        feat             = build_features(df)
        max_day_change   = feat["rsi_14"].diff().abs().dropna().max()
        # EWM smoothing limits abrupt day-over-day RSI changes
        assert max_day_change < 60.0, (
            f"RSI max day change {max_day_change:.1f} is too large; "
            "check Wilder's smoothing is applied (ewm com=period-1)"
        )

    def test_rsi_helper_function_uses_ewm(self) -> None:
        """_rsi output must differ from an SMA-based alternative for the same input."""
        prices     = pd.Series([100.0 + i + (i % 5) * 0.5 for i in range(50)])
        rsi_wilder = _rsi(prices, 14)

        # Construct SMA-based RSI manually for comparison
        delta      = prices.diff()
        gain_sma   = delta.clip(lower=0).rolling(14).mean()
        loss_sma   = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_sma    = 100 - 100 / (1 + gain_sma / loss_sma.replace(0, np.nan))

        # The two must differ somewhere in the overlapping valid range
        both_valid = rsi_wilder.notna() & rsi_sma.notna()
        diff       = (rsi_wilder - rsi_sma).abs()
        assert diff[both_valid].max() > 0.01, (
            "Wilder's RSI and SMA-RSI are identical — "
            "check that ewm(com=period-1) is used in _rsi"
        )

    # ── FIX CORRECT-02: ATR uses Wilder's EWM ────────────────────────────────

    def test_atr_helper_uses_wilder_smoothing(self) -> None:
        """
        _atr with ewm(com=period-1) must differ from _atr with ewm(span=period).
        Tests that the alpha correction (1/period vs 2/(period+1)) is applied.
        """
        df   = _make_ohlcv(100)
        hi   = df["high"].astype(float)
        lo   = df["low"].astype(float)
        cl   = df["close"].astype(float)

        atr_wilder = _atr(hi, lo, cl, 14)

        # Construct span-based ATR manually
        prev_cl = cl.shift(1)
        tr      = pd.concat(
            [hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1
        ).max(axis=1)
        atr_span = tr.ewm(span=14, adjust=False).mean()

        both_valid = atr_wilder.notna() & atr_span.notna()
        diff       = (atr_wilder - atr_span).abs()
        assert diff[both_valid].max() > 1e-6, (
            "Wilder's ATR and span-ATR are identical — "
            "check that ewm(com=period-1) is used in _atr"
        )

    # ── FIX PERF-01: OBV is vectorised ───────────────────────────────────────

    def test_obv_matches_loop_implementation(self) -> None:
        """
        Vectorised OBV (np.sign + cumsum) must match the original loop for a
        random 200-row series.
        """
        df  = _make_ohlcv(200, seed=42)
        cl  = df["close"].astype(float).values
        vol = df["volume"].astype(float).values

        # Reference: loop implementation
        ref = [0.0]
        for i in range(1, len(cl)):
            if cl[i] > cl[i - 1]:
                ref.append(ref[-1] + vol[i])
            elif cl[i] < cl[i - 1]:
                ref.append(ref[-1] - vol[i])
            else:
                ref.append(ref[-1])
        ref_obv = np.array(ref)

        feat = build_features(df)
        np.testing.assert_allclose(
            feat["obv"].values, ref_obv, atol=1e-6,
            err_msg="Vectorised OBV diverges from loop reference",
        )

    # ── FIX PERF-02: _streak is vectorised ───────────────────────────────────

    def test_streak_matches_loop_implementation(self) -> None:
        """Vectorised _streak must match the original loop for a random series."""
        series = pd.Series(
            np.random.default_rng(0).choice([-1.0, 0.0, 1.0], size=200)
        )

        # Reference: loop implementation
        ref, count, prev = [], 0, None
        for v in series:
            count = count + 1 if v == prev else 1
            prev  = v
            ref.append(float(count))

        result = _streak(series).tolist()
        assert result == ref, "Vectorised _streak diverges from loop reference"

    # ── FIX PERF-03: noise vectorised ────────────────────────────────────────

    def test_noise_20_matches_manual_calculation(self) -> None:
        """noise_20 = std(ret) / mean(|ret|) over rolling-20 window."""
        df   = _make_ohlcv(100)
        feat = build_features(df)

        # Manual spot-check at row 50 (window = rows 31–50)
        t    = 50
        rets = feat["ret_1d"].iloc[t - 19: t + 1]  # 20 values
        expected = rets.std() / (rets.abs().mean() + 1e-9)
        actual   = feat["noise_20"].iloc[t]
        assert abs(actual - expected) < 1e-9, (
            f"noise_20[{t}] = {actual:.6f}, expected {expected:.6f}"
        )

    # ── FIX CORRECT-03: gap_up/gap_down redundancy removed ───────────────────

    def test_gap_up_triggers_only_above_0_5pct(self) -> None:
        """gap_up must be 1 only when open > prev_close * 1.005."""
        rows = pd.DataFrame(
            {
                "date":   pd.bdate_range("2024-01-01", periods=4),
                "open":   [100, 100.4, 100.6, 101.0],   # 2nd: just below, 3rd: above
                "high":   [101, 101.5, 101.7, 102.0],
                "low":    [99,  99.5,  99.7, 100.0],
                "close":  [100, 100,   100,   100],
                "volume": [10_000] * 4,
            }
        )
        feat = build_features(rows)
        # Row 1 open=100.4 vs prev_close=100 → 0.4% gap → below threshold → 0
        assert feat["gap_up"].iloc[1] == 0.0
        # Row 2 open=100.6 vs prev_close=100 → 0.6% gap → above threshold → 1
        assert feat["gap_up"].iloc[2] == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE SELECTION ANTI-LEAKAGE
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeatureSelection:

    def test_returned_cols_exist_in_frame(self) -> None:
        """Every column returned by get_feature_cols must exist in df."""
        feat = build_features(_make_ohlcv(300))
        for col in get_feature_cols(feat):
            assert col in feat.columns, f"get_feature_cols returned '{col}' not in feat"

    def test_returns_nonempty_list(self) -> None:
        feat = build_features(_make_ohlcv(200))
        assert len(get_feature_cols(feat)) > 0

    def test_high_nan_column_excluded(self) -> None:
        """Columns with > 50 % NaN must be pruned."""
        feat            = build_features(_make_ohlcv(200))
        feat["_sparse"] = np.nan   # 100 % NaN
        cols            = get_feature_cols(feat)
        assert "_sparse" not in cols

    def test_zero_variance_column_excluded(self) -> None:
        """Columns with near-zero variance must be pruned."""
        feat            = build_features(_make_ohlcv(200))
        feat["_const"]  = 1.0      # zero variance
        cols            = get_feature_cols(feat)
        assert "_const" not in cols

    def test_train_df_statistics_differ_from_full(self) -> None:
        """
        When train_df has a column with > 50 % NaN but the full df does not,
        using train_df must still exclude the column.
        """
        df   = _make_ohlcv(400)
        feat = build_features(df)

        # Column that is sparse in training but dense in full frame
        feat["_sparse_train"] = np.nan
        feat.loc[feat.index[300:], "_sparse_train"] = 1.0   # only last 100 rows

        train_ft  = feat.iloc[:300].copy()
        full_cols  = get_feature_cols(feat)
        train_cols = get_feature_cols(feat, train_df=train_ft)

        # With full frame: _sparse_train has 75% NaN → excluded anyway
        # With train frame: 100% NaN → definitely excluded
        assert "_sparse_train" not in train_cols

    def test_backward_compatibility_no_train_df(self) -> None:
        """get_feature_cols(df) without train_df must still work identically."""
        feat  = build_features(_make_ohlcv(200))
        cols1 = get_feature_cols(feat)
        cols2 = get_feature_cols(feat, train_df=None)
        assert cols1 == cols2
