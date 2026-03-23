#!/usr/bin/env python3
"""
NEPSE Live — Interactive Stock Predictor
=========================================
Automatically fetches live data for ALL NEPSE-listed stocks.
Browse or search the full company list, pick any symbol, and
get a complete technical analysis + price prediction.

Usage:
    python nepse_live.py                          # interactive menu
    python nepse_live.py --symbol NABIL           # direct prediction
    python nepse_live.py --symbol NABIL --predict 10 --years 5
    python nepse_live.py --list                   # print all symbols and exit

Requirements:
    pip install pandas numpy requests colorama tabulate beautifulsoup4 lxml
"""

import argparse
import sys
import os
import warnings
from typing import Optional, Union
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from colorama import Fore, Style, init
from tabulate import tabulate

from fetcher import fetch_company_list, fetch_history

# Reuse all analysis functions from analyze.py
sys.path.insert(0, os.path.dirname(__file__))
from analyze import add_indicators, detect_trend, detect_anomalies, predict_prices, suggest_strategy

init(autoreset=True)


# ─── Display helpers ─────────────────────────────────────────────────────────

def banner():
    lines = [
        "  ███╗   ██╗███████╗██████╗ ███████╗███████╗",
        "  ████╗  ██║██╔════╝██╔══██╗██╔════╝██╔════╝",
        "  ██╔██╗ ██║█████╗  ██████╔╝███████╗█████╗  ",
        "  ██║╚██╗██║██╔══╝  ██╔═══╝ ╚════██║██╔══╝  ",
        "  ██║ ╚████║███████╗██║     ███████║███████╗",
        "  ╚═╝  ╚═══╝╚══════╝╚═╝     ╚══════╝╚══════╝",
        "",
        "   Nepal Stock Exchange — Live Predictor v2.0",
        "   Data: merolagani.com / nepalstock.com.np",
    ]
    print()
    for line in lines:
        print(Fore.CYAN + line)
    print()


def print_header(title: str):
    w = 70
    print()
    print(Fore.CYAN + "═" * w)
    print(Fore.CYAN + f"  {title}")
    print(Fore.CYAN + "═" * w)


def print_section(title: str):
    print()
    print(Fore.YELLOW + f"▶ {title}")
    print(Fore.YELLOW + "─" * 60)


# ─── Company picker ──────────────────────────────────────────────────────────

def search_companies(companies: pd.DataFrame, query: str) -> pd.DataFrame:
    q = query.strip().lower()
    mask = (
        companies["symbol"].str.lower().str.contains(q) |
        companies["name"].str.lower().str.contains(q)   |
        companies["sector"].str.lower().str.contains(q)
    )
    return companies[mask].reset_index(drop=True)


def pick_company_interactive(companies: pd.DataFrame) -> dict:
    """
    Show a searchable company menu.
    Returns dict with symbol, name, sector, id.
    """
    print_section("NEPSE Company Browser")
    print(f"  Total listed companies: {Fore.GREEN}{len(companies)}{Style.RESET_ALL}")
    print("  Type a symbol, company name, or sector to search.")
    print("  Press Enter with empty input to show all.")
    print()

    while True:
        try:
            query = input(f"{Fore.CYAN}  Search > {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!")
            sys.exit(0)

        if query == "":
            results = companies
        else:
            results = search_companies(companies, query)

        if results.empty:
            print(f"  {Fore.RED}No matches for '{query}'. Try again.{Style.RESET_ALL}\n")
            continue

        # Show up to 30 results
        show = results.head(30)
        table_data = [
            [i + 1, r["symbol"], r["name"][:45], r["sector"][:25]]
            for i, (_, r) in enumerate(show.iterrows())
        ]
        print()
        print(tabulate(table_data,
                       headers=["#", "Symbol", "Company Name", "Sector"],
                       tablefmt="rounded_outline"))
        if len(results) > 30:
            print(f"  … and {len(results) - 30} more. Refine your search.")

        print()
        try:
            choice = input(
                f"{Fore.CYAN}  Enter # or symbol (or press Enter to search again) > "
                f"{Style.RESET_ALL}"
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Bye!")
            sys.exit(0)

        if choice == "":
            continue

        # Numeric selection
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(show):
                row = show.iloc[idx]
                return row.to_dict()
            else:
                print(f"  {Fore.RED}Invalid number. Try again.{Style.RESET_ALL}\n")
                continue

        # Direct symbol match
        sym_match = companies[companies["symbol"].str.upper() == choice.upper()]
        if not sym_match.empty:
            return sym_match.iloc[0].to_dict()

        print(f"  {Fore.RED}Not found. Enter a # from the list or an exact symbol.{Style.RESET_ALL}\n")


# ─── Daily Prediction Logging ──────────────────────────────────────────────────

def get_log_filename() -> str:
    """Generate today’s filename matching predictions_log-YYYY-MM-DD.json"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    return f"predictions_log-{today_str}.json"

def load_daily_log() -> dict:
    """Load the daily log safely. If it doesn't exist, roll over the most recent file."""
    file_path = get_log_filename()
    import json
    import glob
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
            
    # Try finding yesterday's/previous file to carry forward history
    files = sorted(glob.glob("predictions_log-*.json"), reverse=True)
    if files:
        try:
            with open(files[0], "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_daily_log(data: dict):
    """Save the dictionary back to the daily JSON file safely."""
    file_path = get_log_filename()
    import json
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"  ⚠ Failed to write prediction log: {e}")

def update_actual_in_memory(data: dict, symbol: str, today_date: str, actual_close: float) -> Optional[dict]:
    """Updates actual_close for a prediction made previously for today."""
    if symbol not in data:
        return None
    updated_entry = None
    for entry in data[symbol]:
        if entry.get("date") == today_date and entry.get("actual_close") is None:
            entry["actual_close"] = float(actual_close)
            pred = entry["predicted_close"]
            entry["error_pct"] = round(((entry["actual_close"] - pred) / (pred + 1e-9)) * 100.0, 2)
            updated_entry = entry
    return updated_entry

def calculate_rolling_accuracy(data: dict, symbol: str) -> Optional[float]:
    """Calculate the average absolute error of the last 7 completed predictions."""
    if symbol not in data:
        return None
    completed = [e["error_pct"] for e in data[symbol] if e.get("error_pct") is not None]
    if not completed:
        return None
    recent_7 = completed[-7:]
    avg_error = sum(abs(e) for e in recent_7) / len(recent_7)
    return round(avg_error, 2)

def append_prediction(symbol: str, prediction_for: str, price: float) -> dict:
    """Append a prediction under the symbol without duplicating, and return mutated data."""
    data = load_daily_log()
    symbol = symbol.upper()
    
    if symbol not in data:
        data[symbol] = []
        
    # Check for duplicates
    for entry in data[symbol]:
        if entry.get("date") == prediction_for:
            return data # Already exists
            
    # Append the new prediction with the old schema
    data[symbol].append({
        "date": prediction_for,
        "predicted_close": round(float(price), 2),
        "actual_close": None,
        "error_pct": None
    })
    
    return data


# ─── Full analysis ───────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, symbol: str, n_predict: int):
    print_header(f"NEPSE LIVE ANALYSIS — {symbol.upper()}")

    print_section("Dataset Summary")
    print(f"  Records      : {len(df)}")
    print(f"  Date Range   : {df['date'].iloc[0].strftime('%Y-%m-%d')}  →  {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Latest Close : NPR {df['close'].iloc[-1]:,.2f}")
    max_cl = df['close'].rolling(min(252, len(df))).max().iloc[-1]
    min_cl = df['close'].rolling(min(252, len(df))).min().iloc[-1]
    print(f"  52-Wk High   : NPR {max_cl:,.2f}")
    print(f"  52-Wk Low    : NPR {min_cl:,.2f}")
    if "volume" in df.columns:
        avg_vol = df["volume"].dropna().tail(20).mean()
        print(f"  Avg Vol (20d): {avg_vol:,.0f}" if not np.isnan(avg_vol) else "  Avg Vol: N/A")

    # Load data temporarily to check for evaluation
    log_data = load_daily_log()
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    latest_dataset_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
    
    updated_entry = None
    if latest_dataset_date == today_str:
        actual_close = float(df["close"].iloc[-1])
        # Update log memory (will save later with new prediction appended)
        updated_entry = update_actual_in_memory(log_data, symbol, today_str, actual_close)
        
        # -----------------------------------------
        if updated_entry is None and len(df) > 20:
            # Predict today's price using data up to yesterday
            bt_pred = predict_prices(df.iloc[:-1].copy(), 1)[0]
            bt_price = bt_pred["predicted_close"]
            
            # Manually inject this as yesterday's prediction into log_data
            if symbol not in log_data:
                log_data[symbol] = []
            backtest_entry = {
                "date": today_str,
                "predicted_close": round(float(bt_price), 2),
                "actual_close": float(actual_close),
                "error_pct": round(((actual_close - bt_price) / (float(bt_price) + 1e-9)) * 100.0, 2)
            }
            log_data[symbol].append(backtest_entry)
            updated_entry = backtest_entry
        # -----------------------------------------

        if updated_entry and updated_entry.get("predicted_close") is not None:
            pred = updated_entry["predicted_close"]
            actual = updated_entry["actual_close"]

            diff = actual - pred
            err_pct = updated_entry["error_pct"]
            abs_err = abs(err_pct)
            
            print_section("Today’s Prediction vs Actual")
            print(f"  Predicted Close (for today): NPR {pred:,.2f}")
            print(f"  Actual Close (today):        NPR {actual:,.2f}")
            color = Fore.GREEN if abs_err < 2 else Fore.YELLOW if abs_err < 5 else Fore.RED
            print(f"  Difference:                  {color}{diff:+,.2f} NPR ({err_pct:+.2f}%){Style.RESET_ALL}")
            print(f"  Prediction Accuracy:         {color}{abs_err:.2f}% abs error{Style.RESET_ALL}")

    df = add_indicators(df)
    trend_info = detect_trend(df)
    
    if updated_entry:
        pred = updated_entry["predicted_close"]
        actual = updated_entry["actual_close"]
        err_pct = updated_entry["error_pct"]
        abs_err = abs(err_pct)
        
        bias = "Accurate"
        if pred > actual: bias = "Overestimated"
        elif pred < actual: bias = "Underestimated"
            
        magnitude = "Low"
        if abs_err > 3: magnitude = "High"
        elif abs_err >= 1: magnitude = "Moderate"
            
        print_section("Daily Prediction Summary")
        print(f"  Model Bias:      {bias}")
        print(f"  Error Magnitude: {magnitude}")
        
        insight = f"Model {bias.lower()} price."
        if bias == "Overestimated" and trend_info['rsi'] < 40:
             insight += " The asset lost momentum faster than expected as RSI drifted lower."
        elif bias == "Underestimated" and trend_info['rsi'] > 60:
             insight += " The asset showed stronger bullish momentum than the model anticipated."
        elif trend_info['macd_bullish'] and bias == "Underestimated":
             insight += " A positive MACD crossover likely fueled extra upside."
        print(f"  Insight:         {insight}")
        
        rolling_err = calculate_rolling_accuracy(log_data, symbol)
        if rolling_err is not None:
            print(f"\n  7-Day Avg Prediction Error: {rolling_err:.2f}%")

    print_section("Trend & Momentum Analysis")
    tc = trend_info["trend_color"]
    print(f"  Overall Trend    : {tc}{trend_info['trend']}{Style.RESET_ALL}  (score {trend_info['score']:+d}/6)")
    print(f"  20-Day Change    : {trend_info['price_change_pct_20d']:+.2f}%")
    print(f"  5-Day Momentum   : {trend_info['recent_5d_momentum']:+.2f}%")
    print(f"  RSI (14)         : {trend_info['rsi']}  →  {trend_info['rsi_label']}")
    print(f"  MACD             : {'📈 Bullish crossover' if trend_info['macd_bullish'] else '📉 Bearish crossover'}")
    print(f"  Bollinger Pos    : {trend_info['bb_position_pct']}%  (0=lower band, 100=upper band)")
    print(f"  Volatility (ATR) : {trend_info['volatility_pct']}%  →  {trend_info['volatility_label']}")
    if trend_info.get("support"):
        print(f"  Support (20d)    : NPR {trend_info['support']:,.2f}")
    if trend_info.get("resistance"):
        print(f"  Resistance (20d) : NPR {trend_info['resistance']:,.2f}")

    anomalies = detect_anomalies(df)
    print_section(f"Anomaly Detection  ({len(anomalies)} flagged in recent data)")
    if anomalies:
        anom_table = [[
            a["date"], f"NPR {a['close']:,.2f}",
            f"{a['change_pct']:+.2f}%", f"{a['z_score']:+.2f}σ", a["label"]
        ] for a in anomalies]
        print(tabulate(anom_table,
                       headers=["Date", "Close", "Change%", "Z-Score", "Type"],
                       tablefmt="simple"))
    else:
        print("  No significant anomalies detected.")

    print_section(f"Price Prediction — Next {n_predict} Trading Sessions")
    try:
        predictions = predict_prices(df, n_predict)
        pred_table = [[
            p["day"], p["date"],
            f"NPR {p['predicted_close']:,.2f}",
            f"NPR {p['low_band']:,.2f}",
            f"NPR {p['high_band']:,.2f}",
            f"{p['change_pct']:+.2f}%",
        ] for p in predictions]
        print(tabulate(pred_table,
                       headers=["Day", "Date", "Predicted", "Low Band", "High Band", "Δ%"],
                       tablefmt="rounded_outline"))
    except ValueError as e:
        print(f"  {Fore.RED}Prediction skipped: {e}")
        predictions = []

    if predictions:
        # Evaluate context update, then append new prediction, then save.
        next_day_pred = predictions[0]
        # Memory is rolled over, updated with evaluation, now append next
        final_data = append_prediction(symbol, next_day_pred["date"], next_day_pred["predicted_close"])
        # Carry over the evaluated current state
        if 'log_data' in locals() and symbol in log_data and updated_entry:
             # Make sure the evaluated entry is preserved inside final_data
             for i, entry in enumerate(final_data[symbol]):
                 if entry.get("date") == updated_entry["date"]:
                     final_data[symbol][i] = updated_entry
        save_daily_log(final_data)
        print(f"\n  ✔ Prediction saved to {get_log_filename()}")
        
        strategies = suggest_strategy(trend_info, predictions, df)
        print_section("Trading Strategy Signals")
        for s in strategies:
            color = (
                Fore.GREEN if "BUY" in s["signal"] else
                Fore.RED   if "SELL" in s["signal"] else
                Fore.YELLOW
            )
            print(f"  {s['emoji']}  [{color}{s['signal']}{Style.RESET_ALL}] ({s['strength']})")
            print(f"       {s['reason']}")

    print_section("Risk Summary")
    risks = []
    if trend_info["volatility_label"] == "HIGH 🔥":
        risks.append("🔥 High volatility — position size carefully, use stop-losses.")
    if trend_info["rsi"] > 70:
        risks.append("⚠️  RSI overbought — correction risk, avoid chasing the rally.")
    if trend_info["rsi"] < 30:
        risks.append("⚠️  RSI oversold — may drop further; wait for reversal confirmation.")
    if len(anomalies) >= 3:
        risks.append(f"⚡ {len(anomalies)} price anomalies detected — possible news-driven moves.")
    if abs(trend_info["recent_5d_momentum"]) < 0.5:
        risks.append("💤 Low momentum — market consolidating; wait for breakout direction.")
    if not risks:
        risks.append("✅ No major risk flags. Maintain regular review of positions.")
    for r in risks:
        print(f"  {r}")

    print_section("How Predictions Work")
    print("  Blends 3 methods (weighted equally based on trend strength):")
    print("  • Linear Regression (40%) — fits long-term price trend line.")
    print("  • Holt Exponential Smoothing (40%) — tracks recent momentum.")
    print("  • Momentum Carry-Forward (20%) — projects avg recent daily return.")
    print("  Confidence bands = ±1.5× recent ATR-scaled volatility.")
    print("  ⚠  Always use stop-losses. Predictions are probabilistic, not guaranteed.")

    print()
    print(Fore.CYAN + "═" * 70)
    print(Fore.CYAN + "  Analysis complete. Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Fore.CYAN + "═" * 70)
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Live — fetch real data for any stock and predict prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nepse_live.py                          # interactive menu
  python nepse_live.py --symbol NABIL           # direct prediction for NABIL
  python nepse_live.py --symbol UPPER --predict 10 --years 5
  python nepse_live.py --list                   # print all NEPSE symbols
        """
    )
    parser.add_argument("--symbol",  help="Stock symbol (e.g. NABIL, NTC, UPPER)")
    parser.add_argument("--predict", type=int, default=7,
                        help="Number of future trading days to predict (5-10, default 7)")
    parser.add_argument("--years",   type=int, default=3,
                        help="Years of historical data to fetch (default 3)")
    parser.add_argument("--list",    action="store_true",
                        help="Print all NEPSE company symbols and exit")
    args = parser.parse_args()

    banner()

    # ── Load company list ────────────────────────────────────────────────────
    print(f"{Fore.YELLOW}  Loading NEPSE company list…{Style.RESET_ALL}")
    companies = fetch_company_list()
    print()

    # ── --list mode ──────────────────────────────────────────────────────────
    if args.list:
        table = [
            [r["symbol"], r["name"][:50], r["sector"]]
            for _, r in companies.iterrows()
        ]
        print(tabulate(table, headers=["Symbol", "Name", "Sector"], tablefmt="rounded_outline"))
        print(f"\n  Total: {len(companies)} companies")
        return

    # ── Determine symbol ─────────────────────────────────────────────────────
    if args.symbol:
        sym_upper = args.symbol.strip().upper()
        match = companies[companies["symbol"].str.upper() == sym_upper]
        if match.empty:
            print(f"  {Fore.YELLOW}Symbol '{sym_upper}' not in company list — attempting fetch anyway.{Style.RESET_ALL}")
            selected = {"symbol": sym_upper, "name": sym_upper, "sector": "", "id": ""}
        else:
            selected = match.iloc[0].to_dict()
    else:
        selected = pick_company_interactive(companies)

    symbol = selected["symbol"].upper()
    cid    = str(selected.get("id", "") or "")
    name   = selected.get("name", symbol)
    sector = selected.get("sector", "")

    print()
    print(f"  {Fore.GREEN}Selected:{Style.RESET_ALL} {symbol}  —  {name}")
    if sector:
        print(f"  Sector  : {sector}")
    print()
    print(f"  {Fore.YELLOW}Fetching {args.years} years of live price data…{Style.RESET_ALL}")

    # ── Fetch history ────────────────────────────────────────────────────────
    try:
        df = fetch_history(symbol, company_id=cid, years=args.years)
    except RuntimeError as e:
        print(f"\n  {Fore.RED}Error: {e}{Style.RESET_ALL}")
        sys.exit(1)

    n = max(5, min(10, args.predict))
    run_analysis(df, symbol, n)

    # ── Predict another? ─────────────────────────────────────────────────────
    if not args.symbol:
        try:
            again = input(
                f"\n{Fore.CYAN}  Analyse another stock? (y/n) > {Style.RESET_ALL}"
            ).strip().lower()
            if again == "y":
                main()
        except (KeyboardInterrupt, EOFError):
            pass


if __name__ == "__main__":
    main()
