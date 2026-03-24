#!/usr/bin/env python3
"""
nepse_live.py — NEPSE Production Predictor v3.0
================================================
Full ML pipeline: fetch → feature engineering → ensemble model →
sentiment analysis → walk-forward backtest → forecast → report.

Usage:
    python nepse_live.py                          # interactive menu
    python nepse_live.py --symbol NABIL           # direct prediction
    python nepse_live.py --symbol NABIL --predict 10 --years 5
    python nepse_live.py --symbol NABIL --backtest # full backtest report
    python nepse_live.py --list                   # list all symbols

Requirements:
    pip install pandas numpy scikit-learn lightgbm xgboost optuna shap
                requests colorama tabulate beautifulsoup4 lxml matplotlib
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tabulate import tabulate

# Local modules
from fetcher import fetch_company_list, fetch_history, get_aggregate_sentiment
from features import build_features, add_targets, get_feature_cols
from models import NEPSEEnsemble, ForecastPoint
from analyze import (
    add_indicators, detect_trend, detect_anomalies,
    predict_prices, suggest_strategy,
    print_header, print_section,
)

init(autoreset=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports")
DATA_DIR   = Path("data")


# ─── Banner ───────────────────────────────────────────────────────────────────

def banner():
    lines = [
        "  ███╗   ██╗███████╗██████╗ ███████╗███████╗",
        "  ████╗  ██║██╔════╝██╔══██╗██╔════╝██╔════╝",
        "  ██╔██╗ ██║█████╗  ██████╔╝███████╗█████╗  ",
        "  ██║╚██╗██║██╔══╝  ██╔═══╝ ╚════██║██╔══╝  ",
        "  ██║ ╚████║███████╗██║     ███████║███████╗",
        "  ╚═╝  ╚═══╝╚══════╝╚═╝     ╚══════╝╚══════╝",
        "",
        "   Nepal Stock Exchange — ML Predictor v3.0",
        "   LightGBM + Random Forest + XGBoost Ensemble",
        "   + News Sentiment + Walk-Forward Backtest",
    ]
    print()
    for line in lines:
        print(Fore.CYAN + line)
    print()


# ─── Company Picker ───────────────────────────────────────────────────────────

def search_companies(companies: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip().lower()
    mask = (
        companies["symbol"].str.lower().str.contains(q) |
        companies["name"].fillna("").str.lower().str.contains(q) |
        companies["sector"].fillna("").str.lower().str.contains(q)
    )
    return companies[mask].reset_index(drop=True)


def pick_company_interactive(companies: pd.DataFrame) -> dict:
    print_section("NEPSE Company Browser")
    print(f"  Total listed: {Fore.GREEN}{len(companies)}{Style.RESET_ALL} companies")
    print("  Search by symbol, name, or sector. Enter to show all.\n")

    while True:
        try:
            query = input(f"{Fore.CYAN}  Search > {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!"); sys.exit(0)

        results = search_companies(companies, query) if query else companies
        if results.empty:
            print(f"  {Fore.RED}No matches. Try again.{Style.RESET_ALL}\n"); continue

        show = results.head(30)
        print()
        print(tabulate(
            [[i+1, r["symbol"], r["name"][:45], r["sector"][:25]]
             for i, (_, r) in enumerate(show.iterrows())],
            headers=["#", "Symbol", "Company", "Sector"],
            tablefmt="rounded_outline",
        ))
        if len(results) > 30:
            print(f"  … and {len(results)-30} more. Refine your search.")
        print()

        try:
            choice = input(f"{Fore.CYAN}  # or symbol (Enter=search again) > {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!"); sys.exit(0)

        if not choice:
            continue
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(show):
                return show.iloc[idx].to_dict()
            continue
        m = companies[companies["symbol"].str.upper() == choice.upper()]
        if not m.empty:
            return m.iloc[0].to_dict()
        print(f"  {Fore.RED}Not found.{Style.RESET_ALL}\n")


# ─── Prediction Log ───────────────────────────────────────────────────────────

def _log_path() -> Path:
    return Path(f"predictions_log-{datetime.now().strftime('%Y-%m-%d')}.json")


def load_log() -> dict:
    import glob
    p = _log_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    files = sorted(glob.glob("predictions_log-*.json"), reverse=True)
    if files:
        try:
            return json.loads(Path(files[0]).read_text())
        except Exception:
            pass
    return {}


def save_log(data: dict):
    try:
        _log_path().write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"  ⚠ Log save failed: {e}")


def append_prediction_to_log(symbol: str, prediction_date: str, price: float) -> dict:
    data = load_log()
    sym = symbol.upper()
    if sym not in data:
        data[sym] = []
    existing = {e.get("date") for e in data[sym]}
    if prediction_date not in existing:
        data[sym].append({"date": prediction_date, "predicted_close": round(price, 2), "actual_close": None, "error_pct": None})
    return data


def evaluate_log(data: dict, symbol: str, actual_date: str, actual_close: float) -> Optional[dict]:
    sym = symbol.upper()
    if sym not in data:
        return None
    for entry in data[sym]:
        if entry.get("date") == actual_date and entry.get("actual_close") is None:
            pred = entry["predicted_close"]
            entry["actual_close"] = round(actual_close, 2)
            entry["error_pct"] = round((actual_close - pred) / (pred + 1e-9) * 100, 2)
            return entry
    return None


def rolling_accuracy(data: dict, symbol: str, n: int = 7) -> Optional[float]:
    sym = symbol.upper()
    completed = [e["error_pct"] for e in data.get(sym, []) if e.get("error_pct") is not None]
    if not completed:
        return None
    return round(np.mean([abs(e) for e in completed[-n:]]), 2)


# ─── Walk-Forward Backtest ────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, symbol: str, window: int = 60, n_test: int = 20) -> dict:
    """
    Rolling-window backtest: train on `window` days, predict 1 step, roll forward.
    Returns summary statistics.
    """
    print_section(f"Walk-Forward Backtest ({n_test} test points, window={window})")
    if len(df) < window + n_test:
        print(f"  {Fore.YELLOW}Insufficient data for backtest (need {window+n_test} rows).{Style.RESET_ALL}")
        return {}

    errors  = []
    dir_acc = []
    results = []

    start_i = len(df) - n_test
    print(f"  Running {n_test} rolling predictions…", end="", flush=True)

    for i in range(n_test):
        train_end = start_i + i
        if train_end < window:
            continue
        train_df  = df.iloc[train_end - window: train_end].copy()
        actual    = float(df["close"].iloc[train_end])
        prev      = float(df["close"].iloc[train_end - 1])
        actual_dt = df["date"].iloc[train_end].strftime("%Y-%m-%d")

        try:
            preds = predict_prices(train_df, 1)
            pred_p = preds[0]["predicted_close"]
            err_pct = (actual - pred_p) / (pred_p + 1e-9) * 100
            correct_dir = int((pred_p > prev) == (actual > prev))
            errors.append(abs(err_pct))
            dir_acc.append(correct_dir)
            results.append({
                "date": actual_dt, "actual": actual,
                "predicted": pred_p, "error_pct": round(err_pct, 2),
                "direction_correct": correct_dir,
            })
        except Exception:
            continue

    print(" done.")

    if not errors:
        print("  No backtest results.")
        return {}

    avg_error = np.mean(errors)
    dir_accuracy = np.mean(dir_acc) * 100
    mape_val = avg_error

    # Print results table (last 10)
    table = [
        [r["date"], f"NPR {r['actual']:,.2f}", f"NPR {r['predicted']:,.2f}",
         f"{r['error_pct']:+.2f}%", "✅" if r["direction_correct"] else "❌"]
        for r in results[-10:]
    ]
    print(tabulate(table, headers=["Date", "Actual", "Predicted", "Error%", "Dir"], tablefmt="simple"))

    summary = {
        "n_predictions": len(errors),
        "avg_abs_error_pct": round(avg_error, 2),
        "directional_accuracy_pct": round(dir_accuracy, 2),
        "mape": round(mape_val, 2),
        "max_error_pct": round(max(errors), 2),
        "within_1pct": round(sum(1 for e in errors if e < 1) / len(errors) * 100, 1),
        "within_2pct": round(sum(1 for e in errors if e < 2) / len(errors) * 100, 1),
        "within_5pct": round(sum(1 for e in errors if e < 5) / len(errors) * 100, 1),
    }

    print()
    print(f"  Avg Abs Error  : {summary['avg_abs_error_pct']:.2f}%")
    print(f"  Directional Acc: {summary['directional_accuracy_pct']:.1f}%")
    print(f"  Within 1%      : {summary['within_1pct']}% of predictions")
    print(f"  Within 2%      : {summary['within_2pct']}% of predictions")
    print(f"  Within 5%      : {summary['within_5pct']}% of predictions")
    print()
    print(f"  {Fore.YELLOW}ℹ  Realistic directional accuracy range for NEPSE: 52–65%{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}ℹ  Price error < 2% on next-day predictions is considered good.{Style.RESET_ALL}")

    return summary


# ─── ML Ensemble Analysis ────────────────────────────────────────────────────

def run_ml_analysis(
    df: pd.DataFrame,
    symbol: str,
    n_predict: int,
    sentiment_score: float = 0.0,
    run_backtest_flag: bool = False,
    use_ml: bool = True,
) -> None:
    print_header(f"NEPSE ML ANALYSIS — {symbol.upper()}")

    # ── Dataset Summary ───────────────────────────────────────────────────────
    print_section("Dataset Summary")
    print(f"  Records      : {len(df)}")
    print(f"  Date Range   : {df['date'].iloc[0].strftime('%Y-%m-%d')}  →  {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Latest Close : NPR {df['close'].iloc[-1]:,.2f}")
    hi52 = df['close'].rolling(min(252, len(df))).max().iloc[-1]
    lo52 = df['close'].rolling(min(252, len(df))).min().iloc[-1]
    print(f"  52-Week High : NPR {hi52:,.2f}")
    print(f"  52-Week Low  : NPR {lo52:,.2f}")
    if "volume" in df.columns and df["volume"].sum() > 0:
        print(f"  Avg Vol (20d): {df['volume'].dropna().tail(20).mean():,.0f}")

    # ── Sentiment ─────────────────────────────────────────────────────────────
    print_section("News Sentiment Analysis")
    print(f"  Fetching news for {symbol}…", end="", flush=True)
    try:
        sentiment = get_aggregate_sentiment(symbol, days_back=7)
        sc = sentiment["score"]
        sentiment_score = sc  # override with fresh value
        label = sentiment["label"]
        count = sentiment["count"]
        color = Fore.GREEN if sc > 0.1 else Fore.RED if sc < -0.1 else Fore.YELLOW
        print(f" found {count} articles.")
        print(f"  Aggregate Score  : {color}{sc:+.3f}{Style.RESET_ALL}  ({label.upper()})")
        if sentiment["articles"]:
            print()
            for art in sentiment["articles"][:3]:
                label_col = Fore.GREEN if art["label"] == "positive" else Fore.RED if art["label"] == "negative" else Fore.WHITE
                print(f"  {label_col}[{art['label'].upper():8s}]{Style.RESET_ALL}  {art['title'][:70]}")
    except Exception as e:
        print(f" unavailable ({e})")
        sentiment = {"score": 0.0, "label": "neutral", "count": 0, "articles": []}

    # ── Technical Indicators ──────────────────────────────────────────────────
    df_ind = add_indicators(df.copy())
    trend_info = detect_trend(df_ind)

    print_section("Trend & Momentum Analysis")
    tc = trend_info["trend_color"]
    print(f"  Overall Trend    : {tc}{trend_info['trend']}{Style.RESET_ALL}  (score {trend_info['score']:+d}/6)")
    print(f"  20-Day Change    : {trend_info['price_change_pct_20d']:+.2f}%")
    print(f"  5-Day Momentum   : {trend_info['recent_5d_momentum']:+.2f}%")
    print(f"  RSI (14)         : {trend_info['rsi']}  →  {trend_info['rsi_label']}")
    print(f"  CCI (20)         : {trend_info['cci']}")
    print(f"  Williams %R      : {trend_info['williams_r']}")
    print(f"  Stoch %K / %D    : {trend_info['stoch_k']} / {trend_info['stoch_d']}")
    print(f"  MACD             : {'📈 Bullish' if trend_info['macd_bullish'] else '📉 Bearish'}")
    print(f"  Bollinger Pos    : {trend_info['bb_position_pct']}%")
    print(f"  Volatility (ATR) : {trend_info['volatility_pct']}%  →  {trend_info['volatility_label']}")
    print(f"  OBV Trend        : {trend_info['obv_trend']}")
    print(f"  Volume Ratio     : {trend_info['vol_ratio']}x")
    if trend_info.get("support"):
        print(f"  Support (20d)    : NPR {trend_info['support']:,.2f}")
    if trend_info.get("resistance"):
        print(f"  Resistance (20d) : NPR {trend_info['resistance']:,.2f}")

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    anomalies = detect_anomalies(df_ind)
    print_section(f"Anomaly Detection  ({len(anomalies)} flagged)")
    if anomalies:
        print(tabulate(
            [[a["date"], f"NPR {a['close']:,.2f}", f"{a['change_pct']:+.2f}%", f"{a['z_score']:+.2f}σ", a["label"]]
             for a in anomalies],
            headers=["Date", "Close", "Change%", "Z-Score", "Type"], tablefmt="simple",
        ))
    else:
        print("  No significant anomalies detected.")

    # ── Optional Backtest ─────────────────────────────────────────────────────
    if run_backtest_flag:
        run_backtest(df, symbol)

    # ── ML Ensemble or Statistical Fallback ──────────────────────────────────
    predictions = []
    ml_report = None
    use_ml_success = False

    if use_ml and len(df) >= 120:
        print_section("ML Ensemble Training")
        print("  Building features & training LightGBM + Random Forest + XGBoost…")
        t0 = time.time()
        try:
            feat_df = build_features(df.copy(), sentiment_score=sentiment_score)
            feat_df = add_targets(feat_df)
            feature_cols = get_feature_cols(feat_df)

            n_opt_trials = 20 if len(df) < 500 else 40
            ensemble = NEPSEEnsemble(
                symbol=symbol,
                n_folds=min(5, max(3, len(df) // 200)),
                optimise=True,
                n_opt_trials=n_opt_trials,
            )
            ensemble.fit(feat_df, feature_cols)
            ml_report = ensemble.report_

            elapsed = time.time() - t0
            print(f"  Training complete in {elapsed:.1f}s")
            print()
            print(f"  {'Fold':<6} {'MAE':>8} {'RMSE':>8} {'Dir Acc':>9} {'MAPE':>8}")
            print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")
            for fold in ml_report.cv_folds:
                da_color = Fore.GREEN if fold.dir_acc >= 58 else Fore.YELLOW if fold.dir_acc >= 52 else Fore.RED
                print(f"  {fold.fold:<6} {fold.mae:>8.2f} {fold.rmse:>8.2f} "
                      f"{da_color}{fold.dir_acc:>8.1f}%{Style.RESET_ALL} {fold.mape:>7.2f}%")
            print()
            da_avg = ml_report.avg_dir_acc
            da_color = Fore.GREEN if da_avg >= 58 else Fore.YELLOW if da_avg >= 52 else Fore.RED
            print(f"  Average: MAE={ml_report.avg_mae:.2f} | RMSE={ml_report.avg_rmse:.2f} | "
                  f"Dir Acc={da_color}{da_avg:.1f}%{Style.RESET_ALL} | MAPE={ml_report.avg_mape:.2f}%")

            # Feature importance
            if ml_report.feature_importance:
                top_feats = list(ml_report.feature_importance.items())[:10]
                print()
                print("  Top 10 Predictive Features:")
                for feat, imp in top_feats:
                    bar = "█" * int(imp * 200)
                    print(f"    {feat:<30} {bar} {imp:.4f}")

            # Forecast
            print_section(f"ML Forecast — Next {n_predict} Trading Sessions")
            ml_predictions = ensemble.forecast(df, feature_cols, horizon=n_predict, sentiment_score=sentiment_score)

            pred_table = []
            for p in ml_predictions:
                dir_color = Fore.GREEN if p.direction_prob > 0.55 else Fore.RED if p.direction_prob < 0.45 else Fore.WHITE
                conf_icon = "🟢" if p.confidence == "high" else "🟡" if p.confidence == "medium" else "🔴"
                pred_table.append([
                    p.day, p.date,
                    f"NPR {p.predicted_close:,.2f}",
                    f"NPR {p.low_band:,.2f}",
                    f"NPR {p.high_band:,.2f}",
                    f"{p.change_pct:+.2f}%",
                    f"{dir_color}{p.direction_prob:.0%}{Style.RESET_ALL}",
                    conf_icon,
                ])
            print(tabulate(
                pred_table,
                headers=["Day", "Date", "Predicted", "Low", "High", "Δ%", "P(Up)", "Conf"],
                tablefmt="rounded_outline",
            ))

            predictions = [
                {
                    "day": p.day, "date": p.date,
                    "predicted_close": p.predicted_close,
                    "low_band": p.low_band, "high_band": p.high_band,
                    "change_pct": p.change_pct,
                    "direction_prob": p.direction_prob,
                    "confidence": p.confidence,
                }
                for p in ml_predictions
            ]
            use_ml_success = True

        except Exception as e:
            print(f"  {Fore.YELLOW}ML training failed ({e}). Falling back to statistical model.{Style.RESET_ALL}")

    if not use_ml_success:
        print_section(f"Statistical Forecast — Next {n_predict} Trading Sessions")
        print(f"  {Fore.YELLOW}(Using blended statistical model — install lightgbm/xgboost for ML){Style.RESET_ALL}")
        try:
            predictions = predict_prices(df, n_predict)
            pred_table = [
                [p["day"], p["date"], f"NPR {p['predicted_close']:,.2f}",
                 f"NPR {p['low_band']:,.2f}", f"NPR {p['high_band']:,.2f}", f"{p['change_pct']:+.2f}%"]
                for p in predictions
            ]
            print(tabulate(pred_table, headers=["Day", "Date", "Predicted", "Low", "High", "Δ%"], tablefmt="rounded_outline"))
        except Exception as e:
            print(f"  {Fore.RED}Prediction failed: {e}")

    # ── Previous prediction evaluation ───────────────────────────────────────
    log_data = load_log()
    latest_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    actual_close = float(df["close"].iloc[-1])

    # Backfill if no prior prediction exists
    if symbol.upper() not in log_data or not any(
        e.get("date") == latest_date for e in log_data.get(symbol.upper(), [])
    ):
        if len(df) > 20:
            try:
                bt_preds = predict_prices(df.iloc[:-1].copy(), 1)
                bt_price = bt_preds[0]["predicted_close"]
                if symbol.upper() not in log_data:
                    log_data[symbol.upper()] = []
                log_data[symbol.upper()].append({
                    "date": latest_date,
                    "predicted_close": round(bt_price, 2),
                    "actual_close": round(actual_close, 2),
                    "error_pct": round((actual_close - bt_price) / (bt_price + 1e-9) * 100, 2),
                })
            except Exception:
                pass

    updated = evaluate_log(log_data, symbol, latest_date, actual_close)

    if updated and updated.get("predicted_close") is not None:
        pred  = updated["predicted_close"]
        act   = updated["actual_close"]
        err   = updated["error_pct"]
        abs_err = abs(err)
        color = Fore.GREEN if abs_err < 2 else Fore.YELLOW if abs_err < 5 else Fore.RED

        print_section("Today's Prediction Accuracy")
        print(f"  Predicted Close : NPR {pred:,.2f}")
        print(f"  Actual Close    : NPR {act:,.2f}")
        print(f"  Error           : {color}{act - pred:+,.2f} NPR ({err:+.2f}%){Style.RESET_ALL}")

        rolling_err = rolling_accuracy(log_data, symbol)
        if rolling_err is not None:
            rc = Fore.GREEN if rolling_err < 2 else Fore.YELLOW if rolling_err < 5 else Fore.RED
            print(f"  7-Day Avg Error : {rc}{rolling_err:.2f}%{Style.RESET_ALL}")

    # ── Trading Signals ───────────────────────────────────────────────────────
    if predictions:
        strategies = suggest_strategy(trend_info, predictions, df_ind)
        print_section("Trading Strategy Signals")
        for s in strategies:
            color = Fore.GREEN if "BUY" in s["signal"] else Fore.RED if "SELL" in s["signal"] else Fore.YELLOW
            print(f"  {s['emoji']}  [{color}{s['signal']}{Style.RESET_ALL}] ({s['strength']})")
            print(f"       {s['reason']}")

        # Save next prediction to log
        next_pred = predictions[0]
        final_log = append_prediction_to_log(symbol, next_pred["date"], next_pred["predicted_close"])
        # Preserve any evaluated entry
        if updated and symbol.upper() in final_log:
            for i, entry in enumerate(final_log[symbol.upper()]):
                if entry.get("date") == updated["date"]:
                    final_log[symbol.upper()][i] = updated
        save_log(final_log)
        print(f"\n  ✔ Prediction for {next_pred['date']} saved to {_log_path().name}")

    # ── Risk Summary ──────────────────────────────────────────────────────────
    print_section("Risk Summary")
    risks = []
    if trend_info["volatility_label"] == "HIGH 🔥":
        risks.append("🔥 High volatility — size positions carefully, set tight stop-losses.")
    if trend_info["rsi"] > 70:
        risks.append("⚠️  RSI overbought — correction risk. Avoid chasing the rally.")
    if trend_info["rsi"] < 30:
        risks.append("⚠️  RSI oversold — may fall further. Wait for reversal confirmation.")
    if len(anomalies) >= 3:
        risks.append(f"⚡ {len(anomalies)} price anomalies — possible news-driven spikes/crashes.")
    if abs(trend_info["recent_5d_momentum"]) < 0.5:
        risks.append("💤 Low momentum — consolidation. Wait for breakout with volume.")
    if sentiment["label"] == "negative" and abs(sentiment["score"]) > 0.2:
        risks.append(f"📰 Negative news sentiment ({sentiment['score']:+.2f}) — watch for downside.")
    if not risks:
        risks.append("✅ No major risk flags. Maintain regular position review.")
    for r in risks:
        print(f"  {r}")

    # ── Save Report ───────────────────────────────────────────────────────────
    if predictions:
        _save_report(symbol, df, trend_info, predictions, sentiment, ml_report)

    print_section("How Predictions Work")
    if use_ml_success:
        print("  ML Ensemble: LightGBM (40%) + Random Forest (25%) + XGBoost (25%) + Ridge (10%)")
        print("  Features: 80+ indicators — price action, momentum, volatility, volume, calendar, sentiment")
        print("  Training: Purged walk-forward CV (no look-ahead bias) + Optuna hyperparameter search")
        print("  Direction: Separate LGB classifier → P(Up) probability per forecast step")
    else:
        print("  Statistical blend: Linear Regression (40%) + Holt Smoothing (40%) + Momentum (20%)")
        print("  Install lightgbm, xgboost, optuna for the full ML ensemble.")
    print("  Circuit breaker: NEPSE ±10% per session enforced on all predictions.")
    print("  ⚠  Predictions are probabilistic, NOT guarantees. Always use stop-losses.")
    print("  ⚠  Realistic directional accuracy: 55–65%. 99% is mathematically impossible.")

    print()
    print(Fore.CYAN + "═" * 72)
    print(Fore.CYAN + "  Analysis complete. Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Fore.CYAN + "═" * 72)
    print()


def _save_report(symbol, df, trend_info, predictions, sentiment, ml_report):
    REPORT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "symbol": symbol.upper(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "last_close": float(df["close"].iloc[-1]),
        "last_date": df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "trend": {k: v for k, v in trend_info.items() if k != "trend_color"},
        "sentiment": {k: v for k, v in sentiment.items() if k != "articles"},
        "forecast": predictions,
        "ml_metrics": {
            "avg_mae": ml_report.avg_mae if ml_report else None,
            "avg_dir_acc": ml_report.avg_dir_acc if ml_report else None,
            "avg_mape": ml_report.avg_mape if ml_report else None,
        } if ml_report else None,
    }
    path = REPORT_DIR / f"{symbol.upper()}_{ts}.json"
    try:
        path.write_text(json.dumps(report, indent=2, default=str))
        logger.debug("Report saved: %s", path)
    except Exception:
        pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Live ML Predictor v3.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nepse_live.py                          # interactive menu
  python nepse_live.py --symbol NABIL           # direct prediction
  python nepse_live.py --symbol UPPER --predict 10 --years 5
  python nepse_live.py --symbol NABIL --backtest  # include backtest
  python nepse_live.py --symbol NABIL --no-ml   # statistical only (fast)
  python nepse_live.py --list                   # list all symbols
        """,
    )
    parser.add_argument("--symbol",   help="Stock symbol (e.g. NABIL, NTC, UPPER)")
    parser.add_argument("--predict",  type=int, default=7, help="Forecast horizon 5-10 (default 7)")
    parser.add_argument("--years",    type=int, default=5, help="Years of history to fetch (default 5)")
    parser.add_argument("--backtest", action="store_true", help="Include walk-forward backtest")
    parser.add_argument("--no-ml",    action="store_true", help="Skip ML training (statistical only, fast)")
    parser.add_argument("--list",     action="store_true", help="Print all NEPSE symbols and exit")
    args = parser.parse_args()

    banner()

    print(f"{Fore.YELLOW}  Loading NEPSE company list…{Style.RESET_ALL}")
    companies = fetch_company_list()
    print()

    if args.list:
        print(tabulate(
            [[r["symbol"], r["name"][:50], r["sector"]] for _, r in companies.iterrows()],
            headers=["Symbol", "Name", "Sector"], tablefmt="rounded_outline",
        ))
        print(f"\n  Total: {len(companies)} companies")
        return

    # Determine symbol
    if args.symbol:
        sym = args.symbol.strip().upper()
        m = companies[companies["symbol"].str.upper() == sym]
        selected = m.iloc[0].to_dict() if not m.empty else {"symbol": sym, "name": sym, "sector": "", "id": ""}
    else:
        selected = pick_company_interactive(companies)

    symbol = selected["symbol"].upper()
    cid    = str(selected.get("id", "") or "")
    name   = selected.get("name", symbol)
    sector = selected.get("sector", "")

    print(f"  {Fore.GREEN}Selected:{Style.RESET_ALL} {symbol}  —  {name}")
    if sector:
        print(f"  Sector  : {sector}")
    print()
    print(f"  {Fore.YELLOW}Fetching {args.years} years of live price data…{Style.RESET_ALL}")

    try:
        df = fetch_history(symbol, company_id=cid, years=args.years)
        print(f"  Loaded {len(df)} rows ({df['date'].iloc[0].strftime('%Y-%m-%d')} to {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    except RuntimeError as e:
        print(f"\n  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    n = max(5, min(10, args.predict))
    run_ml_analysis(
        df=df,
        symbol=symbol,
        n_predict=n,
        run_backtest_flag=args.backtest,
        use_ml=not args.no_ml,
    )

    if not args.symbol:
        try:
            again = input(f"\n{Fore.CYAN}  Analyse another stock? (y/n) > {Style.RESET_ALL}").strip().lower()
            if again == "y":
                main()
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    main()