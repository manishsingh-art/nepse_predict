#!/usr/bin/env python3
"""
nepse_live.py — NEPSE Production Predictor v4.0
================================================
Full ML pipeline: fetch → feature engineering → Nepal calendar →
ensemble model → sentiment analysis → walk-forward backtest → forecast → report.

Usage:
    python nepse_live.py                          # interactive menu
    python nepse_live.py --symbol NABIL           # direct prediction
    python nepse_live.py --symbol NABIL --predict 10 --years 5
    python nepse_live.py --symbol NABIL --backtest
    python nepse_live.py --list
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tabulate import tabulate

from fetcher import fetch_company_list, fetch_history, get_aggregate_sentiment, fetch_live_price, fetch_floorsheet, fetch_news_sentiment
from features import build_features, add_targets, get_feature_cols
from models import NEPSEEnsemble, ForecastPoint
from analyze import (
    add_indicators, detect_trend, detect_anomalies,
    predict_prices, suggest_strategy,
    print_header, print_section,
)
from sector_analysis import identify_market_drivers
from index_predictor import run_index_prediction
from features import add_market_features
from ollama_ai import generate_ai_summary, analyze_sentiment_headlines, is_ollama_available
from fetcher import get_smart_money_signals
from smart_money import SmartMoneyAnalyst
from regime import MarketRegimeDetector
from strategy import TradingStrategyEngine

init(autoreset=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports")
DATA_DIR   = Path("data")

# ── Nepal Calendar ────────────────────────────────────────────────────────────
try:
    from nepal_calendar import (
        NepalMarketCalendar, ad_to_bs, get_bs_month_name,
        next_nepse_trading_dates,
    )
    _NEPAL_CAL = NepalMarketCalendar(fetch_live=True)
    HAS_NEPAL_CAL = True
except ImportError:
    HAS_NEPAL_CAL = False
    _NEPAL_CAL = None


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
        "   Nepal Stock Exchange — ML Predictor v5.0 (Quantum)",
        "   Ensemble: GBM + RF + Ridge + LGB + XGB",
        "   Intelligence: Regime Adaptive + Smart Money + Probability Scenarios",
    ]
    print()
    try:
        for line in lines:
            print(Fore.CYAN + line)
    except UnicodeEncodeError:
        # Some Windows terminals default to a non-UTF8 codepage. Fallback to ASCII banner.
        ascii_lines = [
            "  NEPSE PREDICTOR",
            "  Nepal Stock Exchange — ML Predictor",
            "  Ensemble: GBM + RF + Ridge + (LGB/XGB optional)",
            "  Intelligence: Regime Adaptive + Smart Money + Probabilistic Scenarios",
        ]
        for line in ascii_lines:
            print(line)
    print()


def _print_market_status():
    """Print today's market status using Nepal calendar."""
    if not HAS_NEPAL_CAL or _NEPAL_CAL is None:
        return
    today = date.today()
    bs    = ad_to_bs(today)
    bs_str = f"BS {bs[0]}-{bs[1]:02d}-{bs[2]:02d} ({get_bs_month_name(bs[1])})"

    print_section("Nepal Market Status")
    print(f"  Today (AD)   : {today.strftime('%Y-%m-%d')} ({today.strftime('%A')})")
    print(f"  Today (BS)   : {bs_str}")

    is_trading = _NEPAL_CAL.is_trading_day(today)
    if is_trading:
        market_str = f"{Fore.GREEN}OPEN ✅{Style.RESET_ALL}  (Sun–Thu, 11:00–15:00 NST)"
    else:
        if _NEPAL_CAL.is_weekend(today):
            reason = "Weekend (Fri/Sat closed)"
        elif _NEPAL_CAL.is_public_holiday(today):
            hname = _NEPAL_CAL.get_holiday_name(today) or "Public Holiday"
            reason = f"Public Holiday: {hname}"
        else:
            reason = "Market Closed"
        market_str = f"{Fore.RED}CLOSED ❌{Style.RESET_ALL}  ({reason})"

    print(f"  Market Today : {market_str}")

    # Next trading day
    try:
        next_td = _NEPAL_CAL.next_trading_date(today)
        print(f"  Next Trading : {next_td.strftime('%Y-%m-%d')} ({next_td.strftime('%A')})")
    except Exception:
        pass

    # Upcoming holidays
    try:
        upcoming = _NEPAL_CAL.upcoming_holidays(today, n=3)
        if upcoming:
            hol_str = "  |  ".join(
                f"{h['date']} {h['name']} (in {h['days_away']}d)"
                for h in upcoming
            )
            print(f"  Upcoming     : {hol_str}")
    except Exception:
        pass
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
        try: return json.loads(p.read_text())
        except Exception: return {}
    files = sorted(glob.glob("predictions_log-*.json"), reverse=True)
    if files:
        try: return json.loads(Path(files[0]).read_text())
        except Exception: pass
    return {}


def save_log(data: dict):
    try: _log_path().write_text(json.dumps(data, indent=2))
    except Exception as e: print(f"  ⚠ Log save failed: {e}")


def append_prediction_to_log(symbol: str, prediction_date: str, price: float) -> dict:
    data = load_log()
    sym  = symbol.upper()
    if sym not in data:
        data[sym] = []
    existing = {e.get("date") for e in data[sym]}
    if prediction_date not in existing:
        data[sym].append({
            "date": prediction_date,
            "predicted_close": round(price, 2),
            "actual_close": None, "error_pct": None,
        })
    return data


def evaluate_log(data: dict, symbol: str, actual_date: str, actual_close: float) -> Optional[dict]:
    sym = symbol.upper()
    if sym not in data: return None
    
    # Determine if today is live to decide update policy
    is_live = False
    if HAS_NEPAL_CAL and _NEPAL_CAL:
        now_dt = datetime.now()
        if _NEPAL_CAL.is_trading_day(now_dt.date()) and 11 <= now_dt.hour < 15:
            is_live = True
            
    for entry in data[sym]:
        # Update if not yet set OR if it's today (live update)
        if entry.get("date") == actual_date:
            if entry.get("actual_close") is None or is_live:
                pred = entry["predicted_close"]
                entry["actual_close"] = round(actual_close, 2)
                entry["error_pct"]    = round((actual_close - pred) / (pred + 1e-9) * 100, 2)
                return entry
    return None


def rolling_accuracy(data: dict, symbol: str, n: int = 7) -> Optional[float]:
    sym       = symbol.upper()
    completed = [e["error_pct"] for e in data.get(sym, []) if e.get("error_pct") is not None]
    if not completed: return None
    return round(np.mean([abs(e) for e in completed[-n:]]), 2)


# ─── Walk-Forward Backtest ────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, symbol: str, window: int = 60, n_test: int = 20) -> dict:
    print_section(f"Walk-Forward Backtest ({n_test} test points, window={window})")
    if len(df) < window + n_test:
        print(f"  {Fore.YELLOW}Insufficient data for backtest.{Style.RESET_ALL}")
        return {}

    errors, dir_acc, results = [], [], []
    start_i = len(df) - n_test
    print(f"  Running {n_test} rolling predictions…", end="", flush=True)

    for i in range(n_test):
        train_end = start_i + i
        if train_end < window: continue
        train_df  = df.iloc[train_end - window: train_end].copy()
        actual    = float(df["close"].iloc[train_end])
        prev      = float(df["close"].iloc[train_end - 1])
        actual_dt = df["date"].iloc[train_end].strftime("%Y-%m-%d")
        try:
            preds   = predict_prices(train_df, 1)
            pred_p  = preds[0]["predicted_close"]
            err_pct = (actual - pred_p) / (pred_p + 1e-9) * 100
            correct = int((pred_p > prev) == (actual > prev))
            errors.append(abs(err_pct)); dir_acc.append(correct)
            results.append({"date": actual_dt, "actual": actual,
                            "predicted": pred_p, "error_pct": round(err_pct, 2),
                            "direction_correct": correct})
        except Exception:
            continue

    print(" done.")
    if not errors: return {}

    table = [
        [r["date"], f"NPR {r['actual']:,.2f}", f"NPR {r['predicted']:,.2f}",
         f"{r['error_pct']:+.2f}%", "✅" if r["direction_correct"] else "❌"]
        for r in results[-10:]
    ]
    print(tabulate(table, headers=["Date", "Actual", "Predicted", "Error%", "Dir"], tablefmt="simple"))

    summary = {
        "n_predictions": len(errors),
        "avg_abs_error_pct": round(np.mean(errors), 2),
        "directional_accuracy_pct": round(np.mean(dir_acc) * 100, 2),
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
    print(f"\n  {Fore.YELLOW}ℹ  Realistic NEPSE directional accuracy: 52–65%{Style.RESET_ALL}")
    return summary


# ─── ML Ensemble Analysis ────────────────────────────────────────────────────

def run_ml_analysis(
    df: pd.DataFrame, symbol: str, n_predict: int,
    sentiment_score: float = 0.0,
    run_backtest_flag: bool = False,
    use_ml: bool = True,
    optimise: bool = True,
    n_trials: Optional[int] = None,
    use_ollama: bool = False,
    ollama_model: str = "llama3",
) -> None:

    print_header(f"NEPSE ML ANALYSIS — {symbol.upper()}")
    
    # Initialize v5.0 variables
    sm_report = {"regime": "RETAIL", "buy_concentration": 0, "sell_concentration": 0}
    regime_info = {"regime": "NEUTRAL", "confidence": 0, "color": "white"}
    trade_plan = None

    # ── Dataset Summary ───────────────────────────────────────────────────────
    print_section("Dataset Summary")
    # Latest vs Prev
    ld = df["date"].iloc[-1]
    last_price = df["close"].iloc[-1]
    prev_price = df["close"].iloc[-2] if len(df) > 1 else last_price
    
    is_live_now = False
    if HAS_NEPAL_CAL and _NEPAL_CAL:
        now_dt = datetime.now()
        # NEPSE hours 11:00-15:00 Sun-Thu
        if _NEPAL_CAL.is_trading_day(now_dt.date()) and 11 <= now_dt.hour < 15:
            is_live_now = True
            
    p_label = "Current (LTP)" if is_live_now else "Latest Close"
    
    print(f"  Records      : {len(df)}")
    print(f"  Date Range   : {df['date'].iloc[0].strftime('%Y-%m-%d')}  →  {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  {p_label:14s} : NPR {last_price:,.2f} [{'POS' if last_price >= prev_price else 'NEG'}]")
    print(f"  Prev Close   : NPR {prev_price:,.2f}")
    print(f"  52-Week High : NPR {df['close'].rolling(min(252,len(df))).max().iloc[-1]:,.2f}")
    print(f"  52-Week Low  : NPR {df['close'].rolling(min(252,len(df))).min().iloc[-1]:,.2f}")
    if "volume" in df.columns and df["volume"].sum() > 0:
        print(f"  Avg Vol (20d): {df['volume'].dropna().tail(20).mean():,.0f}")

    # BS date of last record
    if HAS_NEPAL_CAL:
        try:
            ld = df["date"].iloc[-1]
            ld_date = ld.date() if hasattr(ld, "date") else ld
            bs = ad_to_bs(ld_date)
            print(f"  Last Date BS : {bs[0]}-{bs[1]:02d}-{bs[2]:02d} ({get_bs_month_name(bs[1])})")
        except Exception:
            pass

    # ── Fetch News Early for Sentiment & Anomalies ── v6.1 ──────────────────
    news_titles = []
    news_data = fetch_news_sentiment(symbol)
    if not news_data:
        news_data = fetch_news_sentiment("NEPSE")  # fallback to general news

    # ── Sentiment ─────────────────────────────────────────────────────────────
    print_section("News Sentiment Analysis")
    print(f"  Fetching news for {symbol}... found {len(news_data)} articles.")
    for n in news_data[:3]:
        news_titles.append(n['title'].split('\n')[0].strip())

    # If user enabled Ollama but it's not installed/running, gracefully fallback.
    if use_ollama and not is_ollama_available("http://localhost:11434"):
        print(f"  {Fore.YELLOW}Ollama not detected locally — falling back to non-AI sentiment.{Style.RESET_ALL}")
        use_ollama = False
        
    if use_ollama:
        sentiment = analyze_sentiment_headlines(news_titles, model=ollama_model)
        score = sentiment.get("score", 0.0)
        reason = sentiment.get("reason", "No context")
        cat = sentiment.get("category", "General")
        
        s_color = Fore.GREEN if score > 0.3 else Fore.RED if score < -0.3 else Fore.YELLOW
        s_text = "BULLISH" if score > 0.3 else "BEARISH" if score < -0.3 else "NEUTRAL"
        
        print(f"  AI Score ({cat:9s}) : {s_color}{score:+.3f}  ({s_text}){Style.RESET_ALL}")
        print(f"  AI Analysis      : {reason}")
    else:
        # Fallback to basic keyword scoring
        score = 0.0
        bull_words = ['surge', 'profit', 'dividend', 'growth', 'bull', 'positive']
        bear_words = ['loss', 'decline', 'crash', 'bear', 'negative', 'penalty']
        for t in news_titles:
            t_lower = t.lower()
            if any(w in t_lower for w in bull_words): score += 0.5
            if any(w in t_lower for w in bear_words): score -= 0.5
        score = max(-1.0, min(1.0, score / max(1, len(news_titles))))
        sentiment = {"score": score, "reason": "Keyword basic scoring", "category": "General"}
        
        s_color = Fore.GREEN if score > 0.3 else Fore.RED if score < -0.3 else Fore.YELLOW
        print(f"  Aggregate Score  : {s_color}{score:+.3f}  ({'POSITIVE' if score > 0 else 'NEGATIVE' if score < 0 else 'NEUTRAL'}){Style.RESET_ALL}")
        for t in news_titles[:3]:
            prefix = f"{Fore.GREEN}[POSITIVE]{Style.RESET_ALL}" if any(w in t.lower() for w in bull_words) else \
                     f"{Fore.RED}[NEGATIVE]{Style.RESET_ALL}" if any(w in t.lower() for w in bear_words) else \
                     f"{Fore.YELLOW}[NEUTRAL ]{Style.RESET_ALL}"
            print(f"  {prefix}  {t}")

    # ── Market Microstructure (Smart Money) ── v6.0 ───────────────────────────
    print_section("Market Microstructure (Smart Money)")
    print(f"  Fetching latest floorsheet for {symbol}…", end="", flush=True)
    sm_analyst = SmartMoneyAnalyst()
    floorsheet_df = fetch_floorsheet(symbol)
    sm_report = sm_analyst.analyze_floorsheet(floorsheet_df, recent_ohlcv=df)
    
    if sm_report.get("status") != "No data":
        print(" done.")
        sc_color = Fore.GREEN if "ACCUMULATION" in sm_report["regime"] else Fore.RED if "DISTRIBUTION" in sm_report["regime"] else Fore.YELLOW
        if sm_report.get("wash_trading_alert"): sc_color = Fore.RED
        
        print(f"  SM Regime        : {sc_color}{sm_report['regime']}{Style.RESET_ALL}")
        print(f"  Trap Index       : {Fore.CYAN}{sm_report.get('trap_score', 0)}/100{Style.RESET_ALL} (Manipulation Risk)")
        print(f"  Broker HHI       : {sm_report.get('buy_hhi', 0):.0f} Buy / {sm_report.get('sell_hhi', 0):.0f} Sell")
        print(f"  Concentration    : {sm_report['buy_concentration']*100:.1f}% Buy / {sm_report['sell_concentration']*100:.1f}% Sell")
        if sm_report.get("wash_trading_alert"):
            print(f"  {Fore.RED}⚠️ WASH TRADING ALERT: Same broker dominates both Buy & Sell sides.{Style.RESET_ALL}")
        if sm_report.get("hidden_accumulation"):
            print(f"  {Fore.CYAN}🕵️ HIDDEN ACCUMULATION: Price compression with high institution buying.{Style.RESET_ALL}")
    else:
        print(" found no active floorsheet records.")
        sm_report = {"regime": "RETAIL", "buy_concentration": 0, "sell_concentration": 0, "trap_score": 0}


    # ── Technical Indicators ──────────────────────────────────────────────────
    df_ind     = add_indicators(df.copy())
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

    # ── Regime Detection ── v5.0 ──────────────────────────────────────────────
    regime_detector = MarketRegimeDetector()
    regime_info = regime_detector.detect_regime(df_ind, sm_report)
    print_section("Market Regime Classification")
    rc = getattr(Fore, regime_info['color'].upper(), Fore.WHITE)
    print(f"  Detected Regime  : {rc}{regime_info['regime']}{Style.RESET_ALL}")
    print(f"  Confidence       : {regime_info['confidence']*100:.0f}%")
    print(f"  Volatility (20d) : {regime_info['volatility_pct']}%")

    # ── Anomaly Detection ─────────────────────────────────────────────────────
    anomalies = detect_anomalies(df_ind, news_data)
    print_section(f"Anomaly Detection  ({len(anomalies)} flagged)")
    if anomalies:
        table_data = []
        for a in anomalies:
            row = [a["date"], f"NPR {a['close']:,.2f}", f"{a['change_pct']:+.2f}%", f"{a['z_score']:+.2f}σ", a["label"]]
            if any(anom.get("cause") for anom in anomalies):
                row.append(a.get("cause") or "-")
            table_data.append(row)
        
        headers = ["Date", "Close", "Change%", "Z-Score", "Type"]
        if any(anom.get("cause") for anom in anomalies): headers.append("News Cause (Correlation)")
        
        print(tabulate(table_data, headers=headers, tablefmt="simple"))
    else:
        print("  No significant anomalies detected.")
    if run_backtest_flag:
        run_backtest(df, symbol)

    # ── ML Ensemble or Statistical Fallback ──────────────────────────────────
    predictions  = []
    ml_report    = None
    use_ml_success = False

    if use_ml and len(df) >= 100:
        print_section("ML Ensemble Training")
        print("  Building features (incl. Nepal calendar) & training ensemble…")
        t0 = time.time()
        try:
            # ── Market Integration ──
            logger.info("Fetching NEPSE Index for context...")
            try:
                # Use the same years as the stock df for consistency
                days_avail = (df['date'].iloc[-1] - df['date'].iloc[0]).days
                idx_years = max(1, round(days_avail / 365, 1))
                nepse_df = fetch_history("NEPSE", years=idx_years)
                feat_df = add_market_features(df.copy(), nepse_df)
            except Exception as me:
                logger.warning(f"Market features skip: {me}")
                feat_df = df.copy()

            feat_df      = build_features(feat_df, sentiment_score=sentiment_score, smart_money_info=sm_report)
            feat_df      = add_targets(feat_df)
            feature_cols = get_feature_cols(feat_df)
            
            # Dynamic optimization trials
            if n_trials is None:
                n_opt_trials = 10 if len(df) < 500 else 20
            else:
                n_opt_trials = n_trials

            ensemble = NEPSEEnsemble(
                symbol=symbol,
                n_folds=min(5, max(3, len(df) // 180)),
                optimise=optimise,
                n_opt_trials=n_opt_trials,
            )
            ensemble.fit(feat_df, feature_cols)
            ml_report = ensemble.report_

            elapsed = time.time() - t0
            print(f"  Training complete in {elapsed:.1f}s  |  Models: {', '.join(ml_report.models_used)}")
            print()
            print(f"  {'Fold':<6} {'MAE':>8} {'RMSE':>8} {'Dir Acc':>9} {'MAPE':>8}")
            print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")
            for fold in ml_report.cv_folds:
                da_c = Fore.GREEN if fold.dir_acc >= 58 else Fore.YELLOW if fold.dir_acc >= 52 else Fore.RED
                print(f"  {fold.fold:<6} {fold.mae:>8.2f} {fold.rmse:>8.2f} "
                      f"{da_c}{fold.dir_acc:>8.1f}%{Style.RESET_ALL} {fold.mape:>7.2f}%")
            print()
            da_avg = ml_report.avg_dir_acc
            da_c   = Fore.GREEN if da_avg >= 58 else Fore.YELLOW if da_avg >= 52 else Fore.RED
            print(f"  Average: MAE={ml_report.avg_mae:.2f} | RMSE={ml_report.avg_rmse:.2f} | "
                  f"Dir Acc={da_c}{da_avg:.1f}%{Style.RESET_ALL} | MAPE={ml_report.avg_mape:.2f}%")

            try:
                if ml_report.feature_importance:
                    top_feats = list(ml_report.feature_importance.items())[:10]
                    print("\n  Top 10 Predictive Features:")
                    for feat, imp in top_feats:
                        bar = "#" * int(imp * 50)
                        print(f"    {feat:<35} {bar} {imp:.4f}")
            except: pass

            ml_preds = ensemble.forecast(df, feature_cols, horizon=n_predict, sentiment_score=sentiment_score)
            
            print_section(f"ML Forecast — Next {n_predict} Nepal Trading Sessions")
            try:
                pred_table = []
                for p in ml_preds:
                    dir_c = Fore.GREEN if p.direction_prob > 0.55 else Fore.RED if p.direction_prob < 0.45 else Fore.WHITE
                    conf_icon = "GO" if p.confidence == "high" else "OK" if p.confidence == "medium" else "!!"
                    
                    bs_date_str = "-"
                    if HAS_NEPAL_CAL:
                        try:
                            ad = datetime.strptime(p.date, "%Y-%m-%d").date()
                            bs = ad_to_bs(ad)
                            bs_date_str = f"{bs[0]}-{bs[1]:02d}-{bs[2]:02d}"
                        except: pass

                    pred_table.append([
                        p.day, p.date, bs_date_str, p.day_name[:3],
                        f"NPR {p.predicted_close:,.2f}",
                        f"{p.change_pct:+.2f}%",
                        f"{dir_c}{p.direction_prob:.0%}{Style.RESET_ALL}",
                        conf_icon,
                        f"{p.direction_confidence*100:.0f}%",
                        p.trap_score
                    ])
                print(tabulate(pred_table,
                               headers=["D", "Date AD", "Date BS", "Day", "Base", "Δ%", "P(Up)", "Cnf", "D.Conf", "Trap"],
                               tablefmt="simple"))
            except Exception as ue:
                logger.warning(f"Forecast table display error: {ue}")

            # ── Scenario Analysis (v5.0 Upgrade) ──
            if ml_preds:
                print_section("Scenario Analysis (Probabilistic Outcomes)")
                base_f = ml_preds[min(2, len(ml_preds)-1)] # session 3
                print(f"  Bull Case (30%) → Target: NPR {base_f.scenario_bull:,.2f} (+{((base_f.scenario_bull/last_price)-1)*100:+.2f}%)")
                print(f"  Base Case (50%) → Target: NPR {base_f.predicted_close:,.2f} (+{((base_f.predicted_close/last_price)-1)*100:+.2f}%)")
                print(f"  Bear Case (20%) → Target: NPR {base_f.scenario_bear:,.2f} ({((base_f.scenario_bear/last_price)-1)*100:+.2f}%)")

            
            # Signal generation (v6.1) — ATR-aware strategy
            atr_value = trend_info.get("atr", None)  if "trend_info" in dir() else None
            strat_engine = TradingStrategyEngine()
            trade_plan = strat_engine.generate_strategy(last_price, ml_preds, sm_report, regime_info, atr=atr_value)
            
            try:
                # ── Final Decision Engine Integration (v5.0 Upgrade) ──
                m_prob = float(ml_preds[0].direction_prob)
                t_score = float(trend_info["score"])
                s_score = float(sentiment.get("score", 0.0))
                mom_score = float(trend_info["recent_5d_momentum"])
                vol_score = float(trend_info["vol_ratio"])
                
                final_sig, final_conf = ensemble.compute_final_signal(m_prob, t_score, s_score, mom_score, vol_score)
                
                # Model Reliability (v5.0 Upgrade)
                log_data = load_log()
                accuracy = 0.0
                hist_entries = log_data.get(symbol.upper(), [])
                if hist_entries:
                    correct = sum(1 for e in hist_entries if e.get("error_pct") is not None and abs(e["error_pct"]) < 3) # within 3%
                    # Actually directional:
                    # directional_correct = sum(1 for e in hist_entries if ...) 
                    # Use the error/backtest summary if available
                    accuracy = rolling_accuracy(log_data, symbol) or 0.0
                
                print_section("FINAL MARKET VERDICT")
                sig_color = Fore.GREEN if final_sig == "BUY" else Fore.RED if final_sig == "SELL" else Fore.YELLOW
                print(f"  FINAL SIGNAL      : {sig_color}{Style.BRIGHT}{final_sig}{Style.RESET_ALL}")
                print(f"  Confidence        : {final_conf*100:.1f}%")
                
                # Model Reliability
                rel_c = Fore.GREEN if accuracy < 2 else Fore.YELLOW if accuracy < 5 else Fore.RED
                rel_text = f"{100 - accuracy:.1f}%" if accuracy > 0 else "Pending Data"
                print(f"  Model Reliability : {rel_c}{rel_text}{Style.RESET_ALL} (Historical Precision)")
                
                # Reason Construction
                reasons = []
                reasons.append(f"ML Probability: {m_prob:.0%}")
                reasons.append(f"Trend: {'BULLISH' if t_score > 0 else 'BEARISH' if t_score < 0 else 'NEUTRAL'}")
                reasons.append(f"Sentiment: {'POS' if s_score > 0 else 'NEG' if s_score < 0 else 'NEUT'}")
                print(f"  Reason            : {', '.join(reasons)}")

                print_section("Actionable Trade Plan (v6.1 Strategy Layer)")
                ac = Fore.GREEN if "BUY" in trade_plan["action"] else Fore.RED if ("SELL" in trade_plan["action"] or "AVOID" in trade_plan["action"]) else Fore.YELLOW
                
                # Improved Trade Plan (v5.0 Upgrade) - Zones
                sup20 = trend_info.get("support") or last_price * 0.95
                res20 = trend_info.get("resistance") or last_price * 1.05
                
                print(f"  Buy Zone          : NPR {sup20 * 0.98:,.2f} – {sup20 * 1.02:,.2f} (Support ±2%)")
                print(f"  Breakout Buy      : Above NPR {res20:,.2f} (Resistance Validation)")
                print(f"  Target Price (TP) : NPR {trade_plan['take_profit']:,.2f}")
                print(f"  Stop Loss (SL)    : NPR {trade_plan['stop_loss']:,.2f}")
                print(f"  Risk/Reward Ratio : {trade_plan['risk_reward_ratio']:.2f}x")
                print(f"  Position Weight   : {trade_plan['suggested_size_weight']*100:.0f}% of Normal Unit")
            except: pass

            predictions = ml_preds
            use_ml_success = True

        except Exception as e:
            print(f"  {Fore.YELLOW}ML training failed ({e}). Falling back to statistical model.{Style.RESET_ALL}")

    if not use_ml_success:
        print_section(f"Statistical Forecast — Next {n_predict} Sessions")
        print(f"  {Fore.YELLOW}(Statistical blend — install lightgbm/xgboost for ML ensemble){Style.RESET_ALL}")
        try:
            predictions = predict_prices(df, n_predict)
            headers = ["Day", "Date AD", "Date BS", "Day", "Predicted", "Δ%"]
            pred_table = []
            for p in predictions:
                row = [
                    p["day"], p["date"], p.get("bs_date", "-"), p.get("day_name", "")[:3],
                    f"NPR {p['predicted_close']:,.2f}", f"{p['change_pct']:+.2f}%"
                ]
                pred_table.append(row)
            print(tabulate(pred_table, headers=headers, tablefmt="rounded_outline"))
        except Exception as e:
            print(f"  {Fore.RED}Prediction failed: {e}")

    # ── Previous prediction evaluation ───────────────────────────────────────
    log_data     = load_log()
    latest_date  = df["date"].iloc[-1].strftime("%Y-%m-%d")
    actual_close = float(df["close"].iloc[-1])

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
        pred    = updated["predicted_close"]
        act     = updated["actual_close"]
        err     = updated["error_pct"]
        abs_err = abs(err)
        color   = Fore.GREEN if abs_err < 2 else Fore.YELLOW if abs_err < 5 else Fore.RED
        
        # Determine if Live or Final
        is_live = False
        if HAS_NEPAL_CAL and _NEPAL_CAL:
            now_dt = datetime.now()
            # NEPSE hours 11:00-15:00
            if _NEPAL_CAL.is_trading_day(now_dt.date()) and 11 <= now_dt.hour < 15:
                is_live = True

        status_label = f"{Fore.CYAN}(LIVE){Style.RESET_ALL}" if is_live else f"{Fore.MAGENTA}(FINAL CLIP){Style.RESET_ALL}"
        print_section(f"Today's Prediction Accuracy {status_label}")
        print(f"  Target Close    : NPR {pred:,.2f}")
        print(f"  {'Current Price ' if is_live else 'Actual Close  '} : NPR {act:,.2f}")
        print(f"  Deviation       : {color}{act - pred:+,.2f} NPR ({err:+.2f}%){Style.RESET_ALL}")
        rolling_err = rolling_accuracy(log_data, symbol)
        if rolling_err is not None:
            rc = Fore.GREEN if rolling_err < 2 else Fore.YELLOW if rolling_err < 5 else Fore.RED
            print(f"  7-Day Avg Error : {rc}{rolling_err:.2f}%{Style.RESET_ALL}")
            
    # ── Recent Performance Table ─────────────────────────────────────────────
    hist_log = log_data.get(symbol.upper(), [])
    if len(hist_log) > 1:
        print_section("Recent Prediction Performance")
        
        # Determine if today is live
        is_live = False
        if HAS_NEPAL_CAL and _NEPAL_CAL:
            now_dt = datetime.now()
            if _NEPAL_CAL.is_trading_day(now_dt.date()) and 11 <= now_dt.hour < 15:
                is_live = True
                
        perf_table = []
        for entry in hist_log[-5:]:
            if entry.get("actual_close"):
                e_pct = entry.get("error_pct", 0)
                e_c = Fore.GREEN if abs(e_pct) < 2 else Fore.YELLOW if abs(e_pct) < 5 else Fore.RED
                
                date_label = entry.get("date")
                price_type = "Actual"
                if is_live and date_label == latest_date:
                    date_label += f" {Fore.CYAN}(LIVE){Style.RESET_ALL}"
                    price_type = "LTP"
                
                perf_table.append([
                    date_label,
                    f"NPR {entry.get('predicted_close'):,.2f}",
                    f"NPR {entry.get('actual_close'):,.2f}",
                    f"{e_c}{e_pct:+.2f}%{Style.RESET_ALL}"
                ])
        if perf_table:
            print(tabulate(perf_table, headers=["Date AD", "Predicted", "Actual/LTP", "Error%"], tablefmt="simple"))

    # ── Trading Signals (Legacy section removed in v5.0 Upgrade) ──────────

        next_pred  = predictions[0]
        np_date = next_pred.get("date") if isinstance(next_pred, dict) else getattr(next_pred, "date", "")
        np_close = next_pred.get("predicted_close") if isinstance(next_pred, dict) else getattr(next_pred, "predicted_close", 0.0)
        
        final_log  = append_prediction_to_log(symbol, np_date, np_close)
        if updated and symbol.upper() in final_log:
            for i, entry in enumerate(final_log[symbol.upper()]):
                if entry.get("date") == updated["date"]:
                    final_log[symbol.upper()][i] = updated
        save_log(final_log)
        print(f"\n  ✔ Prediction for {np_date} saved to {_log_path().name}")

    # ── Risk Summary ──────────────────────────────────────────────────────────
    print_section("Risk Summary")
    risks = []
    if trend_info["volatility_label"] == "HIGH 🔥":
        risks.append("🔥 High volatility — size positions carefully, set tight stop-losses.")
    if trend_info["rsi"] > 70:
        risks.append("⚠️  RSI overbought — correction risk.")
    if trend_info["rsi"] < 30:
        risks.append("⚠️  RSI oversold — may fall further. Wait for reversal confirmation.")
    if len(anomalies) >= 3:
        risks.append(f"⚡ {len(anomalies)} price anomalies — possible news-driven spikes/crashes.")
    if abs(trend_info["recent_5d_momentum"]) < 0.5:
        risks.append("💤 Low momentum — consolidation phase.")
    if sentiment.get("score", 0.0) < -0.2:
        risks.append(f"📰 Negative news sentiment ({sentiment.get('score', 0):+.2f}) — watch for downside.")
    if HAS_NEPAL_CAL and _NEPAL_CAL is not None:
        today = date.today()
        days_to_hol = _NEPAL_CAL.days_to_next_holiday(today)
        if days_to_hol <= 3:
            hname = _NEPAL_CAL.upcoming_holidays(today, 1)
            hname_str = hname[0]["name"] if hname else "holiday"
            risks.append(f"🗓  {hname_str} in {days_to_hol} days — expect thin liquidity before close.")
    if not risks:
        risks.append("✅ No major risk flags. Maintain regular position review.")
    for r in risks:
        print(f"  {r}")

    # ── AI Analyst Summary (Ollama) ───────────────────────────────────────────
    if use_ollama:
        print_section(f"AI Analyst Summary ({ollama_model})")
        print(f"  {Fore.YELLOW}Generating AI insights...{Style.RESET_ALL}", end="\r")
        ai_data = {
            "current_price": float(df["close"].iloc[-1]),
            "sentiment_label": "negative" if sentiment.get("score", 0) < -0.1 else "positive" if sentiment.get("score", 0) > 0.1 else "neutral",
            "sentiment_score": round(sentiment.get("score", 0.0), 2),
            "regime": regime_info["regime"],
            "smart_money": sm_report["regime"],
            "strategy": trade_plan["action"] if trade_plan else "HOLD",
            "trend": trend_info["trend"],
            "trend_score": trend_info["score"],
            "rsi": trend_info["rsi"],
            "rsi_label": trend_info["rsi_label"],
            "forecast_summary": f"{len(predictions)} sessions, target {predictions[-1].get('predicted_close', 'N/A') if isinstance(predictions[-1], dict) else getattr(predictions[-1], 'predicted_close', 'N/A') if predictions else 'N/A'}",
            "anomalies": [f"{a['date']}: {a['label']}" for a in anomalies[-3:]]
        }
        summary = generate_ai_summary(symbol, ai_data, model=ollama_model)
        if summary:
            print(" " * 50, end="\r") # Clear the generating message
            print(f"  {summary.strip()}")
        else:
            print(f"  {Fore.RED}AI Analyst unavailable (is Ollama running?){Style.RESET_ALL}")

    # ── Save Report ───────────────────────────────────────────────────────────
    if predictions:
        _save_report(symbol, df, trend_info, predictions, sentiment, ml_report, anomalies, 
                     regime=regime_info, smart_money=sm_report, strategy=trade_plan)

    print_section("How Predictions Work")
    if use_ml_success:
        models_str = " + ".join(ml_report.models_used) if ml_report else "GBM + RF + Ridge"
        print(f"  ML Ensemble: {models_str}")
        print("  Features: 109+ indicators — price action, momentum, volatility,")
        print("            volume, Nepal calendar (BS dates, fiscal quarters, festival")
        print("            proximity, dividend season, NRB policy months)")
        print("  Training: Purged walk-forward CV (no look-ahead bias)")
        print("  Forecast: Nepal trading dates only (skips weekends + public holidays)")
    else:
        print("  Statistical blend: Linear Regression (40%) + Holt Smoothing (40%) + Momentum (20%)")
        print("  Install lightgbm, xgboost, optuna for the full ML ensemble.")
    print("  Circuit breaker: NEPSE ±10% per session enforced on all predictions.")
    print("  ⚠  Predictions are probabilistic, NOT guarantees. Always use stop-losses.")
    print("  ⚠  Realistic directional accuracy: 55–65%.")

    print()
    print(Fore.CYAN + "═" * 72)
    print(Fore.CYAN + "  Analysis complete. " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Fore.CYAN + "═" * 72)
    print()


def _save_report(symbol, df, trend_info, predictions, sentiment, ml_report, anomalies, regime=None, smart_money=None, strategy=None):
    REPORT_DIR.mkdir(exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "symbol": symbol.upper(),
        "version": "6.2 Advanced Resilience & Anomaly Integration",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "last_close": float(df["close"].iloc[-1]),
        "last_date":  df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "regime": regime,
        "smart_money": smart_money,
        "strategy": strategy,
        "trend": {k: v for k, v in trend_info.items() if k != "trend_color"},
        "sentiment": {k: v for k, v in sentiment.items() if k != "articles"},
        "forecast": [vars(p) if hasattr(p, '__dict__') else p for p in predictions],
        "anomalies": anomalies,
        "ml_metrics": {
            "avg_mae": ml_report.avg_mae if ml_report else None,
            "avg_dir_acc": ml_report.avg_dir_acc if ml_report else None,
            "avg_mape": ml_report.avg_mape if ml_report else None,
            "models_used": ml_report.models_used if ml_report else [],
        } if ml_report else None,
    }
    path = REPORT_DIR / f"{symbol.upper()}_{ts}.json"
    try: path.write_text(json.dumps(report, indent=2, default=str))
    except Exception: pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Windows console/codepage safety: prefer UTF-8 to avoid crashes on emoji/box-drawing.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="NEPSE Live ML Predictor v4.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nepse_live.py                          # interactive menu
  python nepse_live.py --symbol NABIL           # direct prediction
  python nepse_live.py --symbol UPPER --predict 10 --years 5
  python nepse_live.py --symbol NABIL --backtest
  python nepse_live.py --symbol NABIL --no-ml   # statistical only (fast)
  python nepse_live.py --list
        """,
    )
    parser.add_argument("--symbol",   help="Stock symbol (e.g. NABIL, NTC, UPPER)")
    parser.add_argument("--predict",  type=int, default=7, help="Forecast horizon 5-10 (default 7)")
    parser.add_argument("--years",    type=int, default=2, help="Years of history (default 2)")
    parser.add_argument("--backtest", action="store_true", help="Include walk-forward backtest")
    parser.add_argument("--no-ml",    action="store_true", help="Skip ML training (fast)")
    parser.add_argument("--fast",     action="store_true", help="Fast ML mode (1yr data, no heavy optimization)")
    parser.add_argument("--list",     action="store_true", help="Print all NEPSE symbols and exit")
    parser.add_argument("--ollama",   action="store_true", help="Generate AI summary using local Ollama")
    parser.add_argument("--ollama-model", default="llama3", help="Ollama model name (default: llama3)")
    args = parser.parse_args()

    banner()
    _print_market_status()

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

    if args.symbol:
        sym      = args.symbol.strip().upper()
        m        = companies[companies["symbol"].str.upper() == sym]
        selected = m.iloc[0].to_dict() if not m.empty else {"symbol": sym, "name": sym, "sector": "", "id": ""}
    else:
        print_section("Main Menu")
        print("  1. Market Overview (NEPSE Index + Sector Analysis)")
        print("  2. Individual Stock Prediction")
        print("  3. Exit")
        print()
        try:
            m_choice = input(f"{Fore.CYAN}  Choice > {Style.RESET_ALL}").strip()
            if m_choice == "1":
                identify_market_drivers()
                run_index_prediction()
                input(f"\n{Fore.YELLOW}  Press Enter to return to menu...{Style.RESET_ALL}")
                main(); return
            elif m_choice == "3":
                print("\n  Bye!"); sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!"); sys.exit(0)
            
        selected = pick_company_interactive(companies)

    symbol = selected["symbol"].upper()
    cid    = str(selected.get("id", "") or "")
    name   = selected.get("name", symbol)
    sector = selected.get("sector", "")

    print(f"  {Fore.GREEN}Selected:{Style.RESET_ALL} {symbol}  —  {name}")
    if sector:
        print(f"  Sector  : {sector}")
    print()
    y = args.years
    opt = True
    trials = None
    if args.fast:
        y = 1
        opt = False
        trials = 5
        print(f"  {Fore.CYAN}Fast model enabled (1yr data, minimal optimization){Style.RESET_ALL}")

    print(f"  {Fore.YELLOW}Fetching {y} years of live price data…{Style.RESET_ALL}")

    try:
        df = fetch_history(symbol, company_id=cid, years=y)
        
        # Patch absolute latest price if market is open
        live_price = fetch_live_price(symbol)
        if live_price is not None and live_price > 0:
            now_dt = datetime.now()
            last_row_date = df["date"].iloc[-1].date()
            if last_row_date == now_dt.date():
                if abs(df.loc[df.index[-1], "close"] - live_price) > 0.01:
                    df.loc[df.index[-1], "close"] = live_price
                    df.loc[df.index[-1], "high"]  = float(max(df.loc[df.index[-1], "high"], live_price))
                    df.loc[df.index[-1], "low"]   = float(min(df.loc[df.index[-1], "low"], live_price))
            elif now_dt.hour >= 11:
                is_trading = True
                if HAS_NEPAL_CAL and _NEPAL_CAL:
                    is_trading = _NEPAL_CAL.is_trading_day(now_dt.date())
                if is_trading:
                    new_row = df.iloc[-1:].copy()
                    new_row["date"]   = pd.Timestamp(now_dt.date())
                    new_row["open"]   = df["close"].iloc[-1]
                    new_row["close"]  = float(live_price)
                    new_row["high"]   = float(max(new_row["open"].values[0], live_price))
                    new_row["low"]    = min(new_row["open"].values[0], live_price)
                    new_row["volume"] = 0
                    df = pd.concat([df, new_row], ignore_index=True)

        print(f"  Loaded {len(df)} rows ({df['date'].iloc[0].strftime('%Y-%m-%d')} → {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    except RuntimeError as e:
        print(f"\n  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    n = max(5, min(10, args.predict))
    run_ml_analysis(
        df=df, symbol=symbol, n_predict=n,
        run_backtest_flag=args.backtest,
        use_ml=not args.no_ml,
        optimise=opt,
        n_trials=trials,
        use_ollama=args.ollama,
        ollama_model=args.ollama_model,
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