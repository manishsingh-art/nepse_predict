import unittest

import numpy as np
import pandas as pd

from backtest_engine import BacktestConfig, generate_signals, run_backtest
from features import clean_ohlcv_data
from pipeline import prepare_pipeline_frame, train_model


def make_price_frame(n: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = rng.normal(0.0008, 0.01, size=n)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=n))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, size=n))
    volume = rng.integers(1000, 5000, size=n)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class PipelineConsistencyTests(unittest.TestCase):
    def test_clean_ohlcv_removes_invalid_rows(self):
        raw = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [100, 100, -1, 102],
            "high": [101, 101, 100, 101],
            "low": [99, 99, 98, 103],
            "close": [100, 100, 99, 102],
            "volume": [1000, 1000, 2000, 3000],
        })
        clean = clean_ohlcv_data(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(clean["date"].iloc[0].strftime("%Y-%m-%d"), "2024-01-01")

    def test_prepare_frame_excludes_smart_money_features(self):
        _, feature_frame, feature_cols = prepare_pipeline_frame(make_price_frame())
        self.assertIn("target_ret_1d", feature_frame.columns)
        self.assertTrue(all(not col.startswith("sm_") for col in feature_cols))

    def test_train_model_uses_return_target(self):
        model, _, _, _ = train_model(
            data=make_price_frame(),
            symbol="TEST",
            optimise=False,
            n_folds=3,
        )
        self.assertEqual(model.report_.target_name, "target_ret_1d")

    def test_backtest_returns_trading_metrics(self):
        predictions = pd.DataFrame({
            "signal_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "signal_close": [100.0, 101.0, 102.0, 101.5],
            "target_close": [101.0, 102.0, 101.5, 103.0],
            "predicted_close": [101.0, 102.2, 101.2, 103.1],
            "predicted_return": [0.01, 0.009, -0.003, 0.012],
            "actual_return": [0.01, 0.0099, -0.0049, 0.0148],
            "direction_prob": [0.60, 0.58, 0.45, 0.62],
        })
        signals = generate_signals(predictions, BacktestConfig(max_position_size=0.10))
        result = run_backtest(signals, config=BacktestConfig(max_position_size=0.10))
        self.assertIn("total_return_pct", result.summary)
        self.assertIn("max_drawdown_pct", result.summary)
        self.assertIn("win_rate_pct", result.summary)
        self.assertIn("buy_hold_return_pct", result.summary)
        self.assertIn("alpha_pct", result.summary)
        self.assertEqual(len(result.equity_curve), len(predictions))

    def test_generate_signals_respects_hysteresis_hold_and_cooldown(self):
        predictions = pd.DataFrame({
            "signal_date": pd.bdate_range("2024-01-01", periods=7).strftime("%Y-%m-%d"),
            "trade_date": pd.bdate_range("2024-01-02", periods=7).strftime("%Y-%m-%d"),
            "signal_close": [100.0] * 7,
            "target_close": [101.0] * 7,
            "predicted_close": [101.0] * 7,
            "predicted_return": [0.010, -0.001, 0.001, -0.002, 0.010, 0.010, 0.012],
            "actual_return": [0.0] * 7,
            "direction_prob": [0.80, 0.49, 0.51, 0.48, 0.80, 0.80, 0.82],
        })
        cfg = BacktestConfig(
            max_position_size=0.10,
            fee_rate=0.0,
            slippage_rate=0.0,
            volatility_threshold=None,
            min_signal_strength=0.0,
            min_hold_days=3,
            cooldown_days=2,
            confirmation_period=1,
        )
        signals = generate_signals(predictions, cfg)
        self.assertEqual(signals["signal"].tolist(), [1, 1, 1, 0, 0, 0, 1])
        self.assertTrue(bool(signals.iloc[1]["exit_blocked_by_hold"]))
        self.assertEqual(signals.iloc[3]["decision_reason"], "exit_predicted_return")
        self.assertEqual(signals.iloc[4]["decision_reason"], "cooldown")

    def test_generate_signals_applies_cost_volatility_and_size_filters(self):
        predictions = pd.DataFrame({
            "signal_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "signal_close": [100.0, 100.0, 100.0],
            "target_close": [101.0, 101.0, 101.0],
            "predicted_close": [101.0, 101.0, 101.0],
            "predicted_return": [0.004, 0.008, 0.008],
            "actual_return": [0.0, 0.0, 0.0],
            "direction_prob": [0.80, 0.80, 0.75],
            "atr_pct_14": [0.010, 0.050, 0.010],
        })
        cfg = BacktestConfig(
            max_position_size=0.10,
            fee_rate=0.004,
            slippage_rate=0.001,
            volatility_threshold=0.030,
            min_signal_strength=0.0,
            confirmation_period=1,
        )
        signals = generate_signals(predictions, cfg)
        self.assertEqual(signals["decision_reason"].tolist()[:2], ["cost", "high_volatility"])
        self.assertEqual(int(signals.iloc[2]["signal"]), 1)
        self.assertAlmostEqual(float(signals.iloc[2]["position_weight"]), 0.049505, places=6)
        self.assertAlmostEqual(float(signals.iloc[2]["signal_strength"]), 0.002, places=6)

    def test_run_backtest_enforces_stop_loss_cooldown(self):
        predictions = pd.DataFrame({
            "signal_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "signal_close": [100.0, 98.0, 98.0],
            "target_close": [98.0, 99.0, 100.0],
            "predicted_close": [101.0, 99.0, 100.0],
            "predicted_return": [0.010, 0.010, 0.010],
            "actual_return": [-0.050, 0.010, 0.020],
            "direction_prob": [0.80, 0.80, 0.80],
        })
        cfg = BacktestConfig(
            max_position_size=0.10,
            fee_rate=0.0,
            slippage_rate=0.0,
            volatility_threshold=None,
            min_signal_strength=0.0,
            stop_loss_pct=0.02,
            cooldown_days=2,
            confirmation_period=1,
        )
        signals = generate_signals(predictions, cfg)
        result = run_backtest(signals, config=cfg)
        self.assertGreater(result.equity_curve[0]["actual_position_weight"], 0.0)
        self.assertEqual(result.equity_curve[1]["actual_position_weight"], 0.0)
        self.assertEqual(result.equity_curve[2]["actual_position_weight"], 0.0)
        self.assertEqual(result.trades[0]["exit_reason"], "stop_loss")
        self.assertEqual(result.summary["num_trades"], 1.0)

    def test_generate_signals_requires_confirmation_and_adapts_to_regime(self):
        predictions = pd.DataFrame({
            "signal_date": pd.bdate_range("2024-01-01", periods=6).strftime("%Y-%m-%d"),
            "trade_date": pd.bdate_range("2024-01-02", periods=6).strftime("%Y-%m-%d"),
            "signal_close": [100.0] * 6,
            "target_close": [101.0] * 6,
            "predicted_close": [101.0] * 6,
            "predicted_return": [0.010, 0.011, 0.012, 0.012, 0.012, 0.012],
            "actual_return": [0.0] * 6,
            "direction_prob": [0.80, 0.80, 0.80, 0.80, 0.62, 0.80],
            "atr_pct_14": [0.010, 0.010, 0.010, 0.010, 0.010, 0.050],
            "regime": ["TRENDING", "TRENDING", "SIDEWAYS", "SIDEWAYS", "BEARISH", "TRENDING"],
        })
        cfg = BacktestConfig(
            max_position_size=0.10,
            fee_rate=0.0,
            slippage_rate=0.0,
            min_signal_strength=0.0,
            confirmation_period=2,
            min_hold_days=1,
            cooldown_days=0,
            volatility_threshold=0.030,
            bearish_min_direction_prob=0.70,
            sideways_position_scale=0.50,
        )
        signals = generate_signals(predictions, cfg)
        self.assertEqual(signals["decision_reason"].tolist()[:2], ["confirmation", "confirmed_entry"])
        self.assertAlmostEqual(float(signals.iloc[1]["position_weight"]), 0.059406, places=6)
        self.assertAlmostEqual(float(signals.iloc[2]["position_weight"]), 0.029703, places=6)
        self.assertEqual(signals.iloc[4]["decision_reason"], "threshold")
        self.assertEqual(signals.iloc[5]["decision_reason"], "high_volatility")

    def test_run_backtest_applies_drawdown_scaling_and_logs_trade_diagnostics(self):
        predictions = pd.DataFrame({
            "signal_date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "trade_date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "signal_close": [100.0, 98.0, 97.0, 99.0],
            "target_close": [98.0, 97.0, 99.0, 101.0],
            "predicted_close": [101.0, 99.0, 100.0, 102.0],
            "predicted_return": [0.010, 0.010, 0.010, -0.010],
            "actual_return": [-0.20, -0.10, 0.05, 0.01],
            "direction_prob": [0.80, 0.80, 0.80, 0.40],
            "atr_pct_14": [0.0, 0.0, 0.0, 0.0],
            "regime": ["TRENDING", "TRENDING", "TRENDING", "TRENDING"],
        })
        cfg = BacktestConfig(
            max_position_size=0.10,
            fee_rate=0.0,
            slippage_rate=0.0,
            volatility_threshold=None,
            min_signal_strength=0.0,
            confirmation_period=1,
            min_hold_days=1,
            cooldown_days=0,
            max_drawdown_threshold=0.01,
            drawdown_scaling=0.50,
        )
        signals = generate_signals(predictions, cfg)
        result = run_backtest(signals, config=cfg)
        self.assertAlmostEqual(result.equity_curve[0]["actual_position_weight"], 0.06, places=6)
        self.assertAlmostEqual(result.equity_curve[1]["actual_position_weight"], 0.03, places=6)
        self.assertTrue(bool(result.equity_curve[1]["drawdown_risk_active"]))
        self.assertIn("signal_strength", result.trades[0])
        self.assertIn("regime", result.trades[0])
        self.assertIn("volatility", result.trades[0])
        self.assertIn("entry_reason", result.trades[0])
        self.assertIn("exit_reason", result.trades[0])
        self.assertIn("buy_hold_return_pct", result.summary)
        self.assertIn("alpha_pct", result.summary)


if __name__ == "__main__":
    unittest.main()
