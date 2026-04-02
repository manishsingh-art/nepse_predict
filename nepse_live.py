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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from colorama import Fore, Style, init
from tabulate import tabulate

from prediction_engine import set_global_determinism, compute_sentiment
from fetcher import (
    fetch_company_list,
    fetch_history,
    fetch_live_price,
    fetch_floorsheet,
    fetch_market_live_status,
    fetch_news_sentiment,
    fetch_nepse_symbols,
    get_aggregate_sentiment,
    is_known_symbol_local,
    resolve_symbol,
    get_company_id,
    _fetch_symbols_autosuggest,
    _read_symbols_cache,
    _read_company_ids_cache,
    _write_symbols_cache,
    _write_company_ids_cache,
)
from features import clean_ohlcv_data
from models import NEPSEEnsemble, ForecastPoint
from models import ENSEMBLE_WEIGHTS
from analyze import (
    add_indicators, detect_trend, detect_anomalies,
    suggest_strategy,
    print_header, print_section,
)
from sector_analysis import identify_market_drivers
from index_predictor import run_index_prediction
from ollama_ai import generate_ai_summary, analyze_sentiment_headlines, is_ollama_available
from fetcher import get_smart_money_signals
from smart_money import SmartMoneyAnalyst
from regime import MarketRegimeDetector
from strategy import TradingStrategyEngine
from decision_engine import DecisionInputs, compute_final_decision
from pipeline import run_full_pipeline, train_model
from backtest_engine import BacktestConfig

init(autoreset=True)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports")
DATA_DIR   = Path("data")

# ── Authoritative NEPSE Calendar Facade ───────────────────────────────────────
try:
    from nepse_market_calendar import get_market_calendar

    _MKT_CAL = get_market_calendar()
    HAS_MKT_CAL = True
except Exception:
    _MKT_CAL = None
    HAS_MKT_CAL = False

# ── Nepal Calendar ────────────────────────────────────────────────────────────
try:
    from nepal_calendar import (
        NepalMarketCalendar, get_bs_month_name,
        next_nepse_trading_dates,
    )
    _NEPAL_CAL = NepalMarketCalendar(fetch_live=True)
    HAS_NEPAL_CAL = True
except ImportError:
    HAS_NEPAL_CAL = False
    _NEPAL_CAL = None

# ── AD↔BS conversion (authoritative) ─────────────────────────────────────────
try:
    from nepse_date_utils import ad_to_bs_ymd, validate_ad_bs_mapping
    HAS_BS_UTILS = True
except Exception:
    HAS_BS_UTILS = False
    ad_to_bs_ymd = None
    validate_ad_bs_mapping = None


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


def _format_bs_date(ad_day: date) -> str:
    if not HAS_BS_UTILS:
        return ""
    try:
        y, m, d = ad_to_bs_ymd(ad_day)
        return f"BS {y}-{m:02d}-{d:02d} ({get_bs_month_name(m)})"
    except Exception:
        return ""


def _determine_market_status(today: date, has_live_data: bool, source: str) -> dict:
    if has_live_data:
        return {
            "label": "OPEN",
            "display": f"{Fore.GREEN}OPEN ✅{Style.RESET_ALL}",
            "reason": "Live market feed detected today",
            "source": source,
        }

    is_weekend = bool(_NEPAL_CAL.is_weekend(today)) if HAS_NEPAL_CAL and _NEPAL_CAL is not None else today.weekday() == 5
    if is_weekend:
        return {
            "label": "CLOSED",
            "display": f"{Fore.RED}CLOSED ❌{Style.RESET_ALL}",
            "reason": "Scheduled weekend closure",
            "source": "weekend_rule",
        }

    return {
        "label": "LIKELY OPEN",
        "display": f"{Fore.YELLOW}LIKELY OPEN{Style.RESET_ALL}",
        "reason": "No live feed confirmed; static-holiday fallback intentionally disabled",
        "source": source or "dynamic_fallback",
    }


def _print_market_status():
    """Print today's market status without relying on static holiday closures."""
    today = date.today()
    bs_str = _format_bs_date(today)
    market_live, live_source = fetch_market_live_status(return_source=True)
    status = _determine_market_status(today, market_live, live_source)

    print_section("Nepal Market Status")
    print(f"  Today (AD)   : {today.strftime('%Y-%m-%d')} ({today.strftime('%A')})")
    if bs_str:
        print(f"  Today (BS)   : {bs_str}")
    print(f"  Market Today : {status['display']}  ({status['reason']})")
    print(f"  Data Source  : {status['source']}")

    # Next trading day
    try:
        if HAS_MKT_CAL and _MKT_CAL is not None:
            next_td = today if status["label"] == "OPEN" else _MKT_CAL.next_trading_day(today)
        elif HAS_NEPAL_CAL and _NEPAL_CAL is not None:
            next_td = today if status["label"] == "OPEN" else _NEPAL_CAL.next_trading_date(today)
        else:
            next_td = None
        if next_td is not None:
            print(f"  Next Trading : {next_td.strftime('%Y-%m-%d')} ({next_td.strftime('%A')})")
    except Exception:
        pass
    print()


def _load_companies_with_notice() -> pd.DataFrame:
    print(f"{Fore.YELLOW}  Loading NEPSE company list…{Style.RESET_ALL}")
    companies = fetch_company_list()
    print()
    return companies


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


def recent_bias(data: dict, symbol: str, n: int = 5) -> float:
    """
    Signed bias over last n completed points:
      + => underprediction (actual > predicted)
      - => overprediction (actual < predicted)
    """
    sym = symbol.upper()
    vals = []
    for e in data.get(sym, []):
        if e.get("error_pct") is None:
            continue
        try:
            vals.append(float(e["error_pct"]) / 100.0)
        except Exception:
            continue
    if not vals:
        return 0.0
    return float(np.mean(vals[-n:]))


# ─── Walk-Forward Backtest ────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame, symbol: str, window: int = 60, n_test: int = 20) -> dict:
    print_section("ML Walk-Forward Backtest")
    if len(df) < 120:
        print(f"  {Fore.YELLOW}Insufficient data for ML backtest.{Style.RESET_ALL}")
        return {}

    market_df = None
    try:
        if symbol.upper() != "NEPSE":
            years = max(1, round((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365, 1))
            market_df = fetch_history("NEPSE", years=years)
    except Exception as exc:
        logger.warning("Backtest market features unavailable: %s", exc)

    result = run_full_pipeline(
        data=df,
        symbol=symbol,
        market_data=market_df,
        forecast_horizon=1,
        optimise=False,
        n_folds=min(5, max(3, len(df) // 180)),
        backtest_config=BacktestConfig(),
    )
    summary = result.backtest.summary
    trades = result.backtest.trades[-10:]

    if trades:
        table = [
            [
                t["entry_date"],
                t["exit_date"],
                f"NPR {t['entry_price']:,.2f}",
                f"NPR {t['exit_price']:,.2f}",
                t["holding_days"],
                f"{t['gross_return_pct']:+.2f}%",
                f"{t['net_pnl']:+,.2f}",
            ]
            for t in trades
        ]
        print(tabulate(
            table,
            headers=["Entry", "Exit", "Entry Px", "Exit Px", "Days", "Gross%", "Net PnL"],
            tablefmt="simple",
        ))

    print()
    print(f"  Total Return   : {summary.get('total_return_pct', 0.0):.2f}%")
    print(f"  Sharpe Ratio   : {summary.get('sharpe_ratio', 0.0):.2f}")
    print(f"  Max Drawdown   : {summary.get('max_drawdown_pct', 0.0):.2f}%")
    print(f"  Win Rate       : {summary.get('win_rate_pct', 0.0):.2f}%")
    print(f"  Trades         : {int(summary.get('num_trades', 0))}")
    print(f"  Exposure       : {summary.get('exposure_pct', 0.0):.2f}%")
    return summary


def run_recent_backtest(df: pd.DataFrame, symbol: str, n_days: int = 30) -> dict:
    print_section(f"Recent ML Backtest (last {n_days} sessions)")
    if len(df) < max(120, n_days + 60):
        print(f"  {Fore.YELLOW}Insufficient data for recent ML backtest.{Style.RESET_ALL}")
        return {}

    recent_df = df.tail(max(120, n_days + 60)).copy().reset_index(drop=True)
    return run_backtest(recent_df, symbol)


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
    seed: int = 42,
    debug: bool = False,
    fast_mode: bool = False,
    data_sources: Optional[dict] = None,
    started_at: Optional[float] = None,
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
    if data_sources:
        hist_source = data_sources.get("price_history")
        live_source = data_sources.get("live_price")
        if hist_source:
            print(f"  Data Source  : {hist_source}")
        if live_source:
            print(f"  Live Feed    : {live_source}")

    # BS date of last record
    if HAS_NEPAL_CAL:
        try:
            ld = df["date"].iloc[-1]
            ld_date = ld.date() if hasattr(ld, "date") else ld
            if HAS_BS_UTILS:
                y, m, d = ad_to_bs_ymd(ld_date)
                print(f"  Last Date BS : {y}-{m:02d}-{d:02d} ({get_bs_month_name(m)})")
        except Exception:
            pass

    days_avail = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    idx_years = max(1, round(days_avail / 365, 1))
    market_df = None
    market_source = None
    news_data = []
    news_source = "skipped"
    floorsheet_df = None
    floorsheet_source = "skipped"

    with ThreadPoolExecutor(max_workers=3) as executor:
        market_future = None
        news_future = None
        floorsheet_future = None
        if symbol.upper() != "NEPSE":
            market_future = executor.submit(fetch_history, "NEPSE", years=idx_years, return_source=True)
        if not fast_mode:
            news_future = executor.submit(fetch_news_sentiment, symbol, return_source=True)
            floorsheet_future = executor.submit(fetch_floorsheet, symbol, return_source=True)

        if market_future is not None:
            try:
                market_df, market_source = market_future.result()
            except Exception as exc:
                logger.warning("Market features unavailable: %s", exc)
        if news_future is not None:
            try:
                news_data, news_source = news_future.result()
            except Exception:
                news_data, news_source = [], "unavailable"
        if floorsheet_future is not None:
            try:
                floorsheet_df, floorsheet_source = floorsheet_future.result()
            except Exception:
                floorsheet_df, floorsheet_source = None, "unavailable"

    # ── Fetch News Early for Sentiment & Anomalies ── v6.1 ──────────────────
    news_titles = []
    if not fast_mode and not news_data:
        news_data, news_source = fetch_news_sentiment("NEPSE", return_source=True)  # fallback to general news

    # ── Sentiment ─────────────────────────────────────────────────────────────
    print_section("News Sentiment Analysis")
    if fast_mode:
        sentiment = {"score": 0.0, "reason": "Skipped in --fast mode", "category": "FAST"}
        print(f"  {Fore.CYAN}Skipped in --fast mode for faster startup.{Style.RESET_ALL}")
    else:
        print(f"  Fetching news for {symbol}... found {len(news_data)} articles.")
        print(f"  Source          : {news_source}")
        for n in news_data[:3]:
            news_titles.append(n['title'].split('\n')[0].strip())

        # If user enabled Ollama but it's not installed/running, gracefully fallback.
        if use_ollama and not is_ollama_available("http://localhost:11434"):
            print(f"  {Fore.YELLOW}Ollama not detected locally — falling back to non-AI sentiment.{Style.RESET_ALL}")
            use_ollama = False
            
        sent = compute_sentiment(
            news_titles,
            use_ollama=use_ollama,
            ollama_model=ollama_model,
            analyze_sentiment_headlines_fn=analyze_sentiment_headlines if use_ollama else None,
        )
        score = sent.final_score
        sentiment = {"score": score, "reason": sent.reason, "category": sent.category}
        s_color = Fore.GREEN if score > 0.3 else Fore.RED if score < -0.3 else Fore.YELLOW
        s_text = "BULLISH" if score > 0.3 else "BEARISH" if score < -0.3 else "NEUTRAL"
        if sent.source == "ollama":
            print(f"  AI Score ({sent.category:9s}) : {s_color}{score:+.3f}  ({s_text}){Style.RESET_ALL}")
            if sent.reason:
                print(f"  AI Analysis      : {sent.reason}")
            if abs(sent.final_score - sent.baseline_score) > 1e-6:
                print(f"  Baseline (kw)    : {sent.baseline_score:+.3f}  |  Delta: {(sent.final_score - sent.baseline_score):+.3f}")
        else:
            print(f"  Aggregate Score  : {s_color}{score:+.3f}  ({s_text}){Style.RESET_ALL}")
            for t in news_titles[:3]:
                tl = t.lower()
                prefix = f"{Fore.GREEN}[POSITIVE]{Style.RESET_ALL}" if any(w in tl for w in ['surge','profit','dividend','growth','bull','positive','upgrade','accumulate']) else \
                         f"{Fore.RED}[NEGATIVE]{Style.RESET_ALL}" if any(w in tl for w in ['loss','decline','crash','bear','negative','penalty','downgrade','sell']) else \
                         f"{Fore.YELLOW}[NEUTRAL ]{Style.RESET_ALL}"
                print(f"  {prefix}  {t}")

    # Ensure sentiment score actually feeds the ML feature pipeline.
    sentiment_score = float(sentiment.get("score", 0.0))
    if debug:
        try:
            if not fast_mode:
                print(f"  Debug: sentiment_source={sent.source} baseline={sent.baseline_score:+.3f} final={sent.final_score:+.3f}")
        except Exception:
            pass

    # ── Market Microstructure (Smart Money) ── v6.0 ───────────────────────────
    print_section("Market Microstructure (Smart Money)")
    sm_analyst = SmartMoneyAnalyst()
    if fast_mode:
        print(f"  {Fore.CYAN}Skipped in --fast mode for faster startup.{Style.RESET_ALL}")
        sm_report = {"regime": "RETAIL", "buy_concentration": 0, "sell_concentration": 0, "trap_score": 0}
    else:
        print(f"  Source           : {floorsheet_source}")
        sm_report = sm_analyst.analyze_floorsheet(floorsheet_df, recent_ohlcv=df)
        
        if sm_report.get("status") != "No data":
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
            print("  No active floorsheet records.")
            sm_report = {"regime": "RETAIL", "buy_concentration": 0, "sell_concentration": 0, "trap_score": 0}


    # ── Technical Indicators ──────────────────────────────────────────────────
    df_ind     = add_indicators(clean_ohlcv_data(df.copy()))
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
    print(f"  Confidence       : {regime_info.get('confidence', 0)*100:.0f}%")
    vol_pct = regime_info.get('volatility_pct')
    if vol_pct is not None:
        print(f"  Volatility (20d) : {vol_pct}%")

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
        run_recent_backtest(df, symbol, n_days=30)

    # ── Unified ML Pipeline ──────────────────────────────────────────────────
    predictions  = []
    ml_report    = None
    use_ml_success = False

    if not use_ml:
        print_section("Pipeline Mode")
        print(f"  {Fore.YELLOW}Statistical mode is deprecated. Running the unified ML pipeline instead.{Style.RESET_ALL}")

    if len(df) < 30:
        print_section("ML Ensemble Training")
        print(f"  {Fore.YELLOW}Insufficient data for ML training ({len(df)} rows — need 100+).{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}This stock was recently listed. Check back once more trading days accumulate.{Style.RESET_ALL}")
        print(f"  Above: price snapshot, trend, and available technical indicators.")

    if len(df) >= 100:
        print_section("ML Ensemble Training")
        print("  Building the unified training/prediction pipeline…")
        t0 = time.time()
        try:
            model, clean_df, feat_df, feature_cols = train_model(
                data=df,
                symbol=symbol,
                market_data=market_df,
                optimise=optimise,
                n_folds=min(5, max(3, len(df) // 180)),
                n_opt_trials=10 if n_trials is None and len(df) < 500 else 20 if n_trials is None else n_trials,
                random_state=seed,
            )
            if debug:
                try:
                    print_section("Debug: Feature Snapshot (last row)")
                    snap_cols = [c for c in feature_cols if c in feat_df.columns]
                    snap = feat_df[snap_cols].iloc[-1].to_dict()
                    # show only a concise subset by default
                    keys = sorted(snap.keys())
                    show = keys[:40]
                    for k in show:
                        v = snap.get(k)
                        if v is None:
                            continue
                        try:
                            vv = float(v)
                            if np.isfinite(vv):
                                print(f"  {k:<24s} {vv:>12.6f}")
                            else:
                                print(f"  {k:<24s} {v}")
                        except Exception:
                            print(f"  {k:<24s} {v}")
                    if len(keys) > len(show):
                        print(f"  ... ({len(keys) - len(show)} more features)")
                except Exception as de:
                    logger.warning(f"Debug feature snapshot failed: {de}")
            
            ml_report = model.report_

            elapsed = time.time() - t0
            cache_hint = f"  |  Market ctx: {market_source}" if market_source else ""
            print(f"  Training complete in {elapsed:.1f}s  |  Models: {', '.join(ml_report.models_used)}{cache_hint}")
            print()
            print(f"  {'Fold':<6} {'MAE%':>8} {'RMSE%':>8} {'Dir Acc':>9} {'Sharpe':>8}")
            print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*9} {'─'*8}")
            for fold in ml_report.cv_folds:
                da_c = Fore.GREEN if fold.dir_acc >= 58 else Fore.YELLOW if fold.dir_acc >= 52 else Fore.RED
                print(f"  {fold.fold:<6} {fold.mae:>8.2f} {fold.rmse:>8.2f} "
                      f"{da_c}{fold.dir_acc:>8.1f}%{Style.RESET_ALL} {fold.sharpe:>8.2f}")
            print()
            da_avg = ml_report.avg_dir_acc
            da_c   = Fore.GREEN if da_avg >= 58 else Fore.YELLOW if da_avg >= 52 else Fore.RED
            print(f"  Average: MAE={ml_report.avg_mae:.2f} | RMSE={ml_report.avg_rmse:.2f} | "
                  f"Dir Acc={da_c}{da_avg:.1f}%{Style.RESET_ALL} | Sharpe={ml_report.avg_sharpe:.2f}")

            try:
                if ml_report.feature_importance:
                    top_feats = list(ml_report.feature_importance.items())[:10]
                    print("\n  Top 10 Predictive Features:")
                    for feat, imp in top_feats:
                        bar = "#" * int(imp * 50)
                        print(f"    {feat:<35} {bar} {imp:.4f}")
            except: pass

            ml_preds = model.forecast(clean_df, feature_cols, horizon=n_predict)
            logger.info("ML prediction success")
            
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
                            if HAS_BS_UTILS:
                                validate_ad_bs_mapping(ad)
                                y, m, d = ad_to_bs_ymd(ad)
                                bs_date_str = f"{y}-{m:02d}-{d:02d}"
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

            
            # Signal generation (v6.1) — ATR-aware strategy
            atr_value = trend_info.get("atr", None)  if "trend_info" in dir() else None
            strat_engine = TradingStrategyEngine()
            trade_plan = strat_engine.generate_strategy(last_price, ml_preds, sm_report, regime_info, atr=atr_value)
            
            try:
                print_section("Actionable Trade Plan (v6.1 Strategy Layer)")
                ac = Fore.GREEN if "BUY" in trade_plan["action"] else Fore.RED if ("SELL" in trade_plan["action"] or "AVOID" in trade_plan["action"]) else Fore.YELLOW
                print(f"  Recommended Action : {ac}{trade_plan['action']}{Style.RESET_ALL}")
                print(f"  Entry              : NPR {trade_plan['entry']:,.2f}")
                print(f"  Target Price (TP)  : NPR {trade_plan['take_profit']:,.2f}")
                print(f"  Stop Loss (SL)     : NPR {trade_plan['stop_loss']:,.2f}")
                if trade_plan.get('atr_used'):
                    print(f"  ATR(14)            : NPR {trade_plan['atr_used']:,.2f} (volatility unit)")
                print(f"  Risk/Reward Ratio  : {trade_plan['risk_reward_ratio']:.2f}x")
                print(f"  Trap Index         : {trade_plan.get('trap_index', 0)}/100")
                print(f"  Position Weight    : {trade_plan['suggested_size_weight']*100:.0f}% of Normal Unit")
                print(f"  Rationale          : {trade_plan['reason']}")
            except: pass

            predictions = ml_preds
            use_ml_success = True

        except Exception as e:
            logger.exception("ML training/prediction failed: %s", e)
            print(f"  {Fore.RED}Unified ML pipeline failed: {e}{Style.RESET_ALL}")

    # ── Previous prediction evaluation ───────────────────────────────────────
    log_data     = load_log()
    latest_date  = df["date"].iloc[-1].strftime("%Y-%m-%d")
    actual_close = float(df["close"].iloc[-1])

    if (not fast_mode) and (symbol.upper() not in log_data or not any(
        e.get("date") == latest_date for e in log_data.get(symbol.upper(), [])
    )):
        if len(df) > 100:
            try:
                hist_market_df = None
                if symbol.upper() != "NEPSE":
                    hist_market_df, _ = fetch_history("NEPSE", years=idx_years, return_source=True)
                hist_model, hist_clean, _, hist_cols = train_model(
                    data=df.iloc[:-1].copy(),
                    symbol=symbol,
                    market_data=hist_market_df,
                    optimise=False,
                    n_folds=min(5, max(3, len(df.iloc[:-1]) // 180)),
                    random_state=seed,
                )
                bt_preds = hist_model.forecast(hist_clean, hist_cols, horizon=1)
                bt_price = bt_preds[0].predicted_close
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

    # ── Trading Signals ───────────────────────────────────────────────────────
    if predictions:
        # Error feedback + stability clamp (post-process forecast only)
        try:
            log_data = load_log()
            completed = [
                float(e["error_pct"])
                for e in log_data.get(symbol.upper(), [])
                if e.get("error_pct") is not None
            ]
            recent_error = float(np.mean(completed[-5:])) if completed else 0.0
            if recent_error > 0:
                mult = 0.98
            elif recent_error < 0:
                mult = 1.02
            else:
                mult = 1.0

            for p in predictions:
                if isinstance(p, dict):
                    if "predicted_close" in p and p["predicted_close"] is not None:
                        v = float(p["predicted_close"]) * mult
                        p["predicted_close"] = float(np.clip(v, last_price * 0.90, last_price * 1.10))
                    for k in ("low_band", "high_band"):
                        if k in p and p[k] is not None:
                            p[k] = float(np.clip(float(p[k]) * mult, last_price * 0.90, last_price * 1.10))
                else:
                    v = float(p.predicted_close) * mult
                    p.predicted_close = float(np.clip(v, last_price * 0.90, last_price * 1.10))
                    p.low_band = float(np.clip(float(p.low_band) * mult, last_price * 0.90, last_price * 1.10))
                    p.high_band = float(np.clip(float(p.high_band) * mult, last_price * 0.90, last_price * 1.10))

            if use_ml_success:
                logger.info("ML prediction success")
                logger.info("Model weights used: %s", ENSEMBLE_WEIGHTS)
                p0 = predictions[0]
                p0_val = p0.get("predicted_close") if isinstance(p0, dict) else getattr(p0, "predicted_close", None)
                logger.info("Final adjusted prediction: %s", p0_val)
        except Exception:
            pass

        strategies = suggest_strategy(trend_info, predictions, df_ind)

        # Unified final decision (single signal)
        try:
            p5 = predictions[min(4, len(predictions) - 1)]
            p5_close = p5.get("predicted_close") if isinstance(p5, dict) else getattr(p5, "predicted_close", None)
            exp_5d = (float(p5_close) - float(last_price)) / (float(last_price) + 1e-9) * 100 if p5_close else 0.0
        except Exception:
            exp_5d = 0.0

        p0 = predictions[0]
        dir_prob = p0.get("direction_prob", 0.5) if isinstance(p0, dict) else getattr(p0, "direction_prob", 0.5)

        # Illiquidity proxy from features (last row)
        illiq_flag = 0.0
        try:
            if "illiquid_flag" in df_ind.columns:
                illiq_flag = float(df_ind["illiquid_flag"].iloc[-1])
        except Exception:
            illiq_flag = 0.0

        final = compute_final_decision(DecisionInputs(
            direction_prob=float(dir_prob),
            expected_ret_5d_pct=float(exp_5d),
            regime=str(regime_info.get("regime", "NEUTRAL")),
            regime_confidence=float(regime_info.get("confidence", 0.0)),
            sentiment_score=float(sentiment.get("score", 0.0)),
            trap_score=float(sm_report.get("trap_score", 0.0)),
            volatility_pct=float(trend_info.get("volatility_pct", 0.0)),
            illiquid_flag=float(illiq_flag),
            technical_signals=strategies,
        ))

        print_section("Final Trade Decision (Unified)")
        ac = Fore.GREEN if final.action == "BUY" else Fore.RED if final.action in ("SELL", "AVOID") else Fore.YELLOW
        print(f"  Action      : {ac}{final.action}{Style.RESET_ALL}")
        print(f"  Confidence  : {final.confidence*100:.0f}%")
        print(f"  Rationale   : {final.rationale}")

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
    if started_at is not None:
        print(Fore.CYAN + f"  Execution time: {time.time() - started_at:.2f}s")
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
            "target_name": ml_report.target_name if ml_report else None,
            "avg_mae": ml_report.avg_mae if ml_report else None,
            "avg_dir_acc": ml_report.avg_dir_acc if ml_report else None,
            "avg_sharpe": ml_report.avg_sharpe if ml_report else None,
            "models_used": ml_report.models_used if ml_report else [],
        } if ml_report else None,
    }
    path = REPORT_DIR / f"{symbol.upper()}_{ts}.json"
    try: path.write_text(json.dumps(report, indent=2, default=str))
    except Exception: pass


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    started_at = time.time()
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
  python nepse_live.py --symbol NABIL --no-ml   # deprecated flag (ignored)
  python nepse_live.py --list
        """,
    )
    parser.add_argument("--symbol",   help="Stock symbol (e.g. NABIL, NTC, UPPER)")
    parser.add_argument("--predict",  type=int, default=7, help="Forecast horizon 5-10 (default 7)")
    parser.add_argument("--years",    type=int, default=2, help="Years of history (default 2)")
    parser.add_argument("--backtest", action="store_true", help="Include walk-forward backtest")
    parser.add_argument("--no-ml",    action="store_true", help="Deprecated; the live CLI is ML-only now")
    parser.add_argument("--fast",     action="store_true", help="Fast ML mode (1yr data, no heavy optimization)")
    parser.add_argument("--list",     action="store_true", help="Print all NEPSE symbols and exit")
    parser.add_argument("--ollama",   action="store_true", help="Generate AI summary using local Ollama")
    parser.add_argument("--ollama-model", default="llama3", help="Ollama model name (default: llama3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible results (default: 42)")
    parser.add_argument("--debug", action="store_true", help="Show debug details (features, reasoning, deltas)")
    parser.add_argument("--allow-unknown-symbol", action="store_true", help="Bypass fast local symbol validation and try direct remote fetches")
    args = parser.parse_args()

    # Guard against rare recursion crashes in some ML / pandas paths
    try:
        sys.setrecursionlimit(max(3000, sys.getrecursionlimit()))
    except Exception:
        pass

    set_global_determinism(args.seed)
    banner()
    _print_market_status()

    if args.list:
        companies = _load_companies_with_notice()
        print(tabulate(
            [[r["symbol"], r["name"][:50], r["sector"]] for _, r in companies.iterrows()],
            headers=["Symbol", "Name", "Sector"], tablefmt="rounded_outline",
        ))
        print(f"\n  Total: {len(companies)} companies")
        return

    if args.symbol:
        sym = args.symbol.strip().upper()
        # ── Dynamic symbol resolution (no hardcoded list) ──────────────────────
        # 1. Check AutoSuggest ID + name cache (instant, no network)
        sym_cache = _read_symbols_cache() or {}
        ids_cache = _read_company_ids_cache() or {}
        company_name = sym_cache.get(sym, "")
        company_id   = ids_cache.get(sym, "")

        # 2. If name or ID missing, call the AutoSuggest API live right now
        if not company_name or not company_id:
            print(f"\n  {Fore.YELLOW}Looking up {sym} on NEPSE…{Style.RESET_ALL}")
            try:
                live_syms, live_ids = _fetch_symbols_autosuggest()
                if live_syms:
                    merged_syms = {**sym_cache, **live_syms}
                    merged_ids  = {**ids_cache, **live_ids}
                    _write_symbols_cache(merged_syms)
                    _write_company_ids_cache(merged_ids)
                    company_name = live_syms.get(sym, "")
                    company_id   = live_ids.get(sym, "")
            except Exception as _e:
                logger.debug("AutoSuggest live lookup failed: %s", _e)

        # 3. Report what we found
        if company_name:
            print(f"  {Fore.GREEN}Found:{Style.RESET_ALL} {sym} — {company_name}")
        else:
            import difflib
            all_known = list((_read_symbols_cache() or {}).keys())
            close = difflib.get_close_matches(sym, all_known, n=3, cutoff=0.72)
            if close:
                print(f"\n  {Fore.YELLOW}{sym} not found in current NEPSE listings.")
                print(f"  Did you mean: {', '.join(close)}?{Style.RESET_ALL}")
            else:
                print(f"\n  {Fore.YELLOW}{sym} not found in current NEPSE listings — attempting direct fetch anyway.{Style.RESET_ALL}")

        selected = {"symbol": sym, "name": company_name or sym, "sector": "", "id": company_id}
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

        companies = _load_companies_with_notice()
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
        with ThreadPoolExecutor(max_workers=2) as executor:
            hist_future = executor.submit(
                fetch_history,
                symbol,
                company_id=cid,
                years=y,
                return_source=True,
                allow_company_lookup=True,
            )
            live_future = executor.submit(fetch_live_price, symbol, return_source=True)

            df, price_history_source = hist_future.result()
            live_price, live_price_source = live_future.result()

        is_new_listing = len(df) < 60
        is_ultra_new = len(df) < 30
        if is_ultra_new:
            print(f"  {Fore.YELLOW}VERY NEW LISTING:{Style.RESET_ALL} Only {len(df)} trading rows available. ML training requires 100+ rows.")
            print(f"  {Fore.CYAN}Showing available price data and technical snapshot. ML forecast skipped.{Style.RESET_ALL}")
            opt = False
            trials = 0
        elif is_new_listing:
            print(f"  {Fore.YELLOW}NEW LISTING DETECTED:{Style.RESET_ALL} Only {len(df)} trading rows. Using lighter settings.")
            opt = False
            trials = 3
        
        # Patch absolute latest price if market is open
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
        print(f"  Price Source : {price_history_source}")
        print(f"  Live Source  : {live_price_source}")
    except RuntimeError as e:
        print(f"\n  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Tried: MeroLagani → NepalStock API → ShareSansar{Style.RESET_ALL}")
        print(f"  {Fore.YELLOW}Check the symbol spelling, your internet connection, or try again later.{Style.RESET_ALL}")
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
        seed=args.seed,
        debug=args.debug,
        fast_mode=args.fast,
        data_sources={"price_history": price_history_source, "live_price": live_price_source},
        started_at=started_at,
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