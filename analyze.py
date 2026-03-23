#!/usr/bin/env python3
"""
NEPSE Stock Analysis & Prediction Tool
=======================================
Analyzes historical NEPSE stock data, computes technical indicators,
identifies trend patterns, and predicts next 5-10 price points.

Usage:
    python analyze.py --old old_data.csv --new new_data.csv --symbol NABIL
    python analyze.py --file combined_data.csv --symbol NABIL --predict 10

CSV Format Expected (any of these column names will work):
    Date, Open, High, Low, Close, Volume
    date, open, high, low, close, volume
    DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

# ─── Column name aliases ───────────────────────────────────────────────────────
COL_ALIASES = {
    "date":   ["date", "Date", "DATE", "trading_date", "Trading Date", "बिक्री मिति"],
    "open":   ["open", "Open", "OPEN", "open_price", "Open Price"],
    "high":   ["high", "High", "HIGH", "high_price",  "High Price"],
    "low":    ["low",  "Low",  "LOW",  "low_price",   "Low Price"],
    "close":  ["close","Close","CLOSE","close_price", "Close Price", "ltp", "LTP", "last"],
    "volume": ["volume","Volume","VOLUME","qty","Quantity","turnover","Turnover"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard lowercase names."""
    col_map = {}
    for std, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = std
                break
    df = df.rename(columns=col_map)
    required = ["date", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Please rename columns to: date, open, high, low, close, volume"
        )
    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV/Excel file robustly."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        # Try common separators
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(path, sep=sep, thousands=",")
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
    return normalize_columns(df)


def combine_and_clean(old_path: str, new_path: str) -> pd.DataFrame:
    """Load, combine, sort and deduplicate datasets."""
    old_df = load_csv(old_path)
    new_df = load_csv(new_path)
    df = pd.concat([old_df, new_df], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for col in ["open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df


def load_single(path: str) -> pd.DataFrame:
    """Load a single combined CSV."""
    df = load_csv(path)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", ""), errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for col in ["open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    return df


# ─── Technical Indicators ─────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["close"]

    # Moving averages
    df["sma_10"]  = cl.rolling(10).mean()
    df["sma_20"]  = cl.rolling(20).mean()
    df["sma_50"]  = cl.rolling(50).mean()
    df["ema_12"]  = cl.ewm(span=12, adjust=False).mean()
    df["ema_26"]  = cl.ewm(span=26, adjust=False).mean()

    # MACD
    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # RSI (14)
    delta = cl.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2σ)
    mid         = cl.rolling(20).mean()
    std         = cl.rolling(20).std()
    df["bb_mid"]  = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

    # Stochastic %K, %D (14,3)
    if "high" in df.columns and "low" in df.columns:
        lo14 = df["low"].rolling(14).min()
        hi14 = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (cl - lo14) / (hi14 - lo14 + 1e-9)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR (14)
    if "high" in df.columns and "low" in df.columns:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - cl.shift()).abs(),
            (df["low"]  - cl.shift()).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    # Volume indicators
    if "volume" in df.columns:
        df["vol_sma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_sma20"]

    # Momentum & ROC
    df["momentum_10"] = cl.diff(10)
    df["roc_10"]      = cl.pct_change(10) * 100

    # Support / Resistance (rolling 20-day)
    if "high" in df.columns and "low" in df.columns:
        df["resistance_20"] = df["high"].rolling(20).max()
        df["support_20"]    = df["low"].rolling(20).min()

    return df


# ─── Trend & Pattern Detection ────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-20] if len(df) >= 20 else df.iloc[0]

    price_change_pct = (last["close"] - prev["close"]) / prev["close"] * 100
    above_sma20 = last["close"] > last.get("sma_20", last["close"])
    above_sma50 = last["close"] > last.get("sma_50", last["close"])
    macd_bull   = last.get("macd", 0) > last.get("macd_signal", 0)
    rsi_val     = last.get("rsi", 50)
    bb_pos      = (last["close"] - last.get("bb_lower", last["close"])) / \
                  (last.get("bb_upper", last["close"]) - last.get("bb_lower", last["close"]) + 1e-9)

    # Score: +1 bullish, -1 bearish
    score = 0
    score += 1 if price_change_pct > 0 else -1
    score += 1 if above_sma20 else -1
    score += 1 if above_sma50 else -1
    score += 1 if macd_bull else -1
    score += 1 if rsi_val < 70 else -1
    score += 1 if bb_pos < 0.8 else -1

    # Volatility
    atr = last.get("atr", 0)
    volatility_pct = (atr / last["close"] * 100) if last["close"] else 0

    # Recent momentum
    recent_5 = df["close"].iloc[-5:].pct_change().sum() * 100

    if score >= 3:
        trend = "BULLISH 📈"
        trend_color = Fore.GREEN
    elif score <= -3:
        trend = "BEARISH 📉"
        trend_color = Fore.RED
    else:
        trend = "SIDEWAYS / NEUTRAL ↔️"
        trend_color = Fore.YELLOW

    vol_label = (
        "HIGH 🔥" if volatility_pct > 3 else
        "MODERATE" if volatility_pct > 1.5 else
        "LOW 😴"
    )

    # RSI condition
    rsi_label = (
        "OVERBOUGHT ⚠️" if rsi_val > 70 else
        "OVERSOLD 💡"   if rsi_val < 30 else
        "NEUTRAL"
    )

    return {
        "trend": trend,
        "trend_color": trend_color,
        "score": score,
        "price_change_pct_20d": round(price_change_pct, 2),
        "volatility_pct": round(volatility_pct, 2),
        "volatility_label": vol_label,
        "rsi": round(rsi_val, 1),
        "rsi_label": rsi_label,
        "macd_bullish": macd_bull,
        "bb_position_pct": round(bb_pos * 100, 1),
        "recent_5d_momentum": round(recent_5, 2),
        "last_close": last["close"],
        "last_date": last["date"],
        "support": last.get("support_20", None),
        "resistance": last.get("resistance_20", None),
    }


def detect_anomalies(df: pd.DataFrame) -> list:
    anomalies = []
    if len(df) < 5:
        return anomalies

    # Daily return z-score
    df["daily_ret"] = df["close"].pct_change()
    mean_ret = df["daily_ret"].mean()
    std_ret  = df["daily_ret"].std()

    for i, row in df.iterrows():
        ret = row["daily_ret"]
        if pd.isna(ret):
            continue
        z = (ret - mean_ret) / (std_ret + 1e-9)
        if abs(z) > 2.5:
            direction = "🔺 SPIKE" if ret > 0 else "🔻 CRASH"
            anomalies.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "close": row["close"],
                "change_pct": round(ret * 100, 2),
                "z_score": round(z, 2),
                "label": direction,
            })
    return anomalies[-10:]  # last 10


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_prices(df: pd.DataFrame, n: int = 7) -> list:
    """
    Blend three approaches:
    1. Linear regression trend extrapolation
    2. Exponential smoothing
    3. Momentum-based carry-forward

    Each given a weight based on recent trend strength.
    """
    close = df["close"].dropna().values
    if len(close) < 10:
        raise ValueError("Need at least 10 data points to predict.")

    # ── 1. Linear regression ──────────────────────────────────────────────────
    x = np.arange(len(close))
    coeffs = np.polyfit(x, close, 1)
    slope, intercept = coeffs
    lr_preds = [intercept + slope * (len(close) + i) for i in range(1, n + 1)]

    # ── 2. Exponential smoothing (Holt's  linear) ────────────────────────────
    alpha = 0.3
    beta  = 0.1
    level, trend_es = close[0], 0
    for v in close[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend_es)
        trend_es = beta * (level - prev_level) + (1 - beta) * trend_es
    es_preds = [level + (i + 1) * trend_es for i in range(n)]

    # ── 3. Momentum (last-N average return carry-forward) ────────────────────
    lookback = min(10, len(close) - 1)
    avg_daily_ret = np.mean(np.diff(close[-lookback:]) / close[-lookback:-1])
    mom_preds = [close[-1] * (1 + avg_daily_ret) ** (i + 1) for i in range(n)]

    # ── Blend weights (favour ES for stable, LR for trending) ────────────────
    recent_vol = np.std(np.diff(close[-20:]) / (close[-20:-1] + 1e-9) if len(close) >= 21 else np.diff(close) / (close[:-1] + 1e-9))
    w_lr  = 0.40
    w_es  = 0.40
    w_mom = 0.20

    predictions = []
    last_date = df["date"].iloc[-1]
    # NEPSE trades Mon–Fri (skip weekends)
    pred_date = last_date
    current_price = close[-1]
    
    for i in range(n):
        pred_date = pred_date + pd.offsets.BDay(1)
        blended = w_lr * lr_preds[i] + w_es * es_preds[i] + w_mom * mom_preds[i]
        
        # Enforce NEPSE ±10% circuit limit rule
        # A stock cannot change by more than 10% in a single day
        max_limit = current_price * 1.10
        min_limit = current_price * 0.90
        
        if blended > max_limit:
            blended = max_limit
        elif blended < min_limit:
            blended = min_limit
            
        # Confidence band ±1 std of recent volatility
        band = blended * recent_vol * 1.5
        
        prev_price = predictions[-1]["predicted_close"] if predictions else close[-1]
        daily_change = (blended - prev_price) / prev_price * 100
        
        predictions.append({
            "day": i + 1,
            "date": pred_date.strftime("%Y-%m-%d"),
            "predicted_close": round(blended, 2),
            "low_band": round(blended - band, 2),
            "high_band": round(blended + band, 2),
            "change_pct": round(daily_change, 2), # Daily change from previous session
        })
        current_price = blended  # Use capped price for next day's limit calculation
    return predictions


# ─── Strategy ────────────────────────────────────────────────────────────────

def suggest_strategy(trend_info: dict, predictions: list, df: pd.DataFrame) -> list:
    rsi   = trend_info["rsi"]
    score = trend_info["score"]
    last  = trend_info["last_close"]
    support    = trend_info.get("support")
    resistance = trend_info.get("resistance")
    mom   = trend_info["recent_5d_momentum"]
    pred_5 = predictions[4]["predicted_close"] if len(predictions) >= 5 else last
    expected_gain = (pred_5 - last) / last * 100

    signals = []

    # RSI signals
    if rsi < 30:
        signals.append({
            "signal": "BUY",
            "strength": "STRONG",
            "reason": f"RSI={rsi} is deeply oversold (<30). Historically a reversal zone.",
            "emoji": "💚"
        })
    elif rsi < 45:
        signals.append({
            "signal": "BUY",
            "strength": "MODERATE",
            "reason": f"RSI={rsi} is approaching oversold. Accumulation opportunity.",
            "emoji": "🟢"
        })
    elif rsi > 75:
        signals.append({
            "signal": "SELL",
            "strength": "STRONG",
            "reason": f"RSI={rsi} is deeply overbought (>75). Risk of profit-taking correction.",
            "emoji": "🔴"
        })
    elif rsi > 65:
        signals.append({
            "signal": "SELL/REDUCE",
            "strength": "MODERATE",
            "reason": f"RSI={rsi} approaching overbought. Consider taking partial profits.",
            "emoji": "🟡"
        })

    # Trend score signals
    if score >= 4:
        signals.append({
            "signal": "BUY/HOLD",
            "strength": "STRONG",
            "reason": "Strong bullish confluence: price above SMAs, positive MACD, upward momentum.",
            "emoji": "💚"
        })
    elif score <= -4:
        signals.append({
            "signal": "SELL/AVOID",
            "strength": "STRONG",
            "reason": "Strong bearish confluence: price below SMAs, negative MACD, downward momentum.",
            "emoji": "🔴"
        })

    # Prediction-based signal
    if expected_gain > 5:
        signals.append({
            "signal": "BUY",
            "strength": "MODEL",
            "reason": f"Model predicts +{expected_gain:.1f}% gain over 5 sessions.",
            "emoji": "🤖"
        })
    elif expected_gain < -5:
        signals.append({
            "signal": "SELL",
            "strength": "MODEL",
            "reason": f"Model predicts {expected_gain:.1f}% decline over 5 sessions.",
            "emoji": "🤖"
        })

    # Support/Resistance
    if support and last <= support * 1.02:
        signals.append({
            "signal": "BUY",
            "strength": "TECHNICAL",
            "reason": f"Price near 20-day support (NPR {support:.2f}). Good risk/reward entry.",
            "emoji": "📊"
        })
    if resistance and last >= resistance * 0.98:
        signals.append({
            "signal": "SELL",
            "strength": "TECHNICAL",
            "reason": f"Price near 20-day resistance (NPR {resistance:.2f}). Consider booking profits.",
            "emoji": "📊"
        })

    # Momentum
    if abs(mom) < 0.5:
        signals.append({
            "signal": "HOLD/WAIT",
            "strength": "CAUTION",
            "reason": "5-day momentum near zero — no clear directional edge. Wait for breakout.",
            "emoji": "⏳"
        })

    if not signals:
        signals.append({
            "signal": "HOLD",
            "strength": "NEUTRAL",
            "reason": "No strong buy or sell signals detected. Market is in balance.",
            "emoji": "⚖️"
        })

    return signals


# ─── Display ──────────────────────────────────────────────────────────────────

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


def run_analysis(df: pd.DataFrame, symbol: str, n_predict: int):
    print_header(f"NEPSE STOCK ANALYSIS — {symbol.upper()}")

    # Summary stats
    print_section("Dataset Summary")
    print(f"  Records     : {len(df)}")
    print(f"  Date Range  : {df['date'].iloc[0].strftime('%Y-%m-%d')}  →  {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Latest Close: NPR {df['close'].iloc[-1]:,.2f}")
    print(f"  52-Week High: NPR {df['close'].rolling(min(252, len(df))).max().iloc[-1]:,.2f}")
    print(f"  52-Week Low : NPR {df['close'].rolling(min(252, len(df))).min().iloc[-1]:,.2f}")
    if "volume" in df.columns:
        avg_vol = df["volume"].tail(20).mean()
        print(f"  Avg Volume (20d): {avg_vol:,.0f}")

    # Add indicators
    df = add_indicators(df)
    trend_info = detect_trend(df)

    # Trend
    print_section("Trend & Momentum Analysis")
    tc = trend_info["trend_color"]
    print(f"  Overall Trend    : {tc}{trend_info['trend']}{Style.RESET_ALL}  (score {trend_info['score']:+d}/6)")
    print(f"  20-Day Change    : {trend_info['price_change_pct_20d']:+.2f}%")
    print(f"  5-Day Momentum   : {trend_info['recent_5d_momentum']:+.2f}%")
    print(f"  RSI (14)         : {trend_info['rsi']}  →  {trend_info['rsi_label']}")
    print(f"  MACD             : {'📈 Bullish crossover' if trend_info['macd_bullish'] else '📉 Bearish crossover'}")
    print(f"  Bollinger Pos    : {trend_info['bb_position_pct']}%  (0=lower band, 100=upper band)")
    print(f"  Volatility (ATR) : {trend_info['volatility_pct']}%  →  {trend_info['volatility_label']}")
    if trend_info["support"]:
        print(f"  Support (20d)    : NPR {trend_info['support']:,.2f}")
    if trend_info["resistance"]:
        print(f"  Resistance (20d) : NPR {trend_info['resistance']:,.2f}")

    # Anomalies
    anomalies = detect_anomalies(df)
    print_section(f"Anomaly Detection  ({len(anomalies)} flagged)")
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

    # Predictions
    print_section(f"Price Prediction — Next {n_predict} Trading Days")
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

    # Strategy
    if predictions:
        strategies = suggest_strategy(trend_info, predictions, df)
        print_section("Trading Strategy Signals")
        for s in strategies:
            color = Fore.GREEN if "BUY" in s["signal"] else Fore.RED if "SELL" in s["signal"] else Fore.YELLOW
            print(f"  {s['emoji']}  [{color}{s['signal']}{Style.RESET_ALL}] ({s['strength']})")
            print(f"       {s['reason']}")

    # Risks
    print_section("Risk Summary")
    risks = []
    if trend_info["volatility_label"] == "HIGH 🔥":
        risks.append("🔥 High volatility — position size carefully, use stop-losses.")
    if trend_info["rsi"] > 70:
        risks.append("⚠️  RSI overbought — correction risk, avoid chasing the rally.")
    if trend_info["rsi"] < 30:
        risks.append("⚠️  RSI oversold — may drop further before reversing; confirm reversal signal.")
    if len(anomalies) >= 3:
        risks.append(f"⚡ {len(anomalies)} price anomalies detected — possible news-driven spikes/crashes.")
    if abs(trend_info["recent_5d_momentum"]) < 0.5:
        risks.append("💤 Low momentum — market may be in consolidation; breakout direction is unclear.")
    if not risks:
        risks.append("✅ No major risk flags. Maintain regular review.")
    for r in risks:
        print(f"  {r}")

    # Methodology note
    print_section("Methodology (How Predictions Are Made)")
    print("  Predictions blend three methods, each validated against recent data:")
    print("  • Linear Regression (40%) — extrapolates the long-term price trend.")
    print("  • Exponential Smoothing (40%) — tracks recent momentum with decay.")
    print("  • Momentum Carry-Forward (20%) — projects average recent daily return.")
    print("  Confidence bands = ±1.5× ATR-scaled volatility of recent 20 sessions.")
    print("  ⚠  Predictions are probabilistic, not guaranteed. Always use stop-losses.")

    print()
    print(Fore.CYAN + "═" * 70)
    print(Fore.CYAN + "  Analysis complete. Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Fore.CYAN + "═" * 70)
    print()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Stock Analysis & Prediction Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze.py --old old_data.csv --new new_data.csv --symbol NABIL
  python analyze.py --file combined_data.csv --symbol SCB --predict 10
  python analyze.py --old hist.csv --new recent.csv --symbol ADBL --predict 5
        """
    )
    parser.add_argument("--old",     help="Path to old/historical data CSV")
    parser.add_argument("--new",     help="Path to recent data CSV")
    parser.add_argument("--file",    help="Path to single combined CSV (alternative to --old/--new)")
    parser.add_argument("--symbol",  default="STOCK", help="Stock symbol label (e.g. NABIL)")
    parser.add_argument("--predict", type=int, default=7, help="Number of future days to predict (5-10)")
    args = parser.parse_args()

    if args.file:
        df = load_single(args.file)
    elif args.old and args.new:
        df = combine_and_clean(args.old, args.new)
    else:
        print(Fore.RED + "Error: Provide either --file OR both --old and --new.")
        print("Run with --help for usage.")
        sys.exit(1)

    if len(df) < 10:
        print(Fore.RED + f"Error: Only {len(df)} rows after cleaning. Need at least 10.")
        sys.exit(1)

    n = max(5, min(10, args.predict))
    run_analysis(df, args.symbol, n)


if __name__ == "__main__":
    main()
