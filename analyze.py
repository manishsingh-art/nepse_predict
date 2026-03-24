#!/usr/bin/env python3
"""
analyze.py — NEPSE Technical Analysis Engine (v3.0)
====================================================
Standalone technical analysis usable with CSV files.
Retained for backward compatibility + standalone use.

Usage:
    python analyze.py --old old_data.csv --new new_data.csv --symbol NABIL
    python analyze.py --file combined_data.csv --symbol NABIL --predict 10
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)

COL_ALIASES = {
    "date":   ["date", "Date", "DATE", "trading_date", "Trading Date"],
    "open":   ["open", "Open", "OPEN", "open_price", "Open Price"],
    "high":   ["high", "High", "HIGH", "high_price", "High Price"],
    "low":    ["low", "Low", "LOW", "low_price", "Low Price"],
    "close":  ["close", "Close", "CLOSE", "close_price", "Close Price", "ltp", "LTP", "last"],
    "volume": ["volume", "Volume", "VOLUME", "qty", "Quantity", "turnover", "Turnover"],
}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for std, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = std
                break
    df = df.rename(columns=col_map)
    for req in ["date", "close"]:
        if req not in df.columns:
            raise ValueError(f"Missing column '{req}'. Found: {list(df.columns)}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        for sep in [",", "\t", ";"]:
            try:
                df = pd.read_csv(path, sep=sep, thousands=",")
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
    return normalize_columns(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", ""), errors="coerce")
    for col in ["open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    # Forward-fill missing OHLC from close
    for col in ["open", "high", "low"]:
        if col not in df.columns:
            df[col] = df["close"]
        else:
            df[col] = df[col].fillna(df["close"])
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df


def combine_and_clean(old_path: str, new_path: str) -> pd.DataFrame:
    return _clean(pd.concat([load_csv(old_path), load_csv(new_path)], ignore_index=True))


def load_single(path: str) -> pd.DataFrame:
    return _clean(load_csv(path))


# ─── Technical Indicators ─────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    cl = df["close"].astype(float)
    hi = df["high"].astype(float) if "high" in df.columns else cl
    lo = df["low"].astype(float)  if "low"  in df.columns else cl
    vol = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)

    # Moving averages
    for w in [5, 10, 20, 50, 100]:
        df[f"sma_{w}"] = cl.rolling(w).mean()
        df[f"ema_{w}"] = cl.ewm(span=w, adjust=False).mean()

    # MACD
    ema12 = cl.ewm(span=12, adjust=False).mean()
    ema26 = cl.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # RSI
    for p in [7, 14, 21]:
        delta = cl.diff()
        gain  = delta.clip(lower=0).rolling(p).mean()
        loss  = (-delta.clip(upper=0)).rolling(p).mean()
        rs    = gain / loss.replace(0, np.nan)
        df[f"rsi_{p}"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi_14"]  # backward-compat alias

    # Bollinger Bands (20)
    mid = cl.rolling(20).mean()
    std = cl.rolling(20).std()
    df["bb_mid"]   = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-9)
    df["bb_pos"]   = (cl - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # Stochastic
    lo14 = lo.rolling(14).min()
    hi14 = hi.rolling(14).max()
    df["stoch_k"] = 100 * (cl - lo14) / (hi14 - lo14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    # CCI
    tp = (hi + lo + cl) / 3
    df["cci"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)

    # Williams %R
    df["williams_r"] = -100 * (hi.rolling(14).max() - cl) / (hi.rolling(14).max() - lo.rolling(14).min() + 1e-9)

    # OBV
    obv = [0.0]
    for i in range(1, len(df)):
        obv.append(obv[-1] + (vol.iloc[i] if cl.iloc[i] > cl.iloc[i-1] else -vol.iloc[i] if cl.iloc[i] < cl.iloc[i-1] else 0))
    df["obv"] = obv

    # Volume
    df["vol_sma20"] = vol.rolling(20).mean()
    df["vol_ratio"] = vol / (df["vol_sma20"] + 1e-9)

    # Momentum / ROC
    df["roc_10"]      = cl.pct_change(10) * 100
    df["momentum_10"] = cl.diff(10)
    df["ret_1d"]      = cl.pct_change(fill_method=None)

    # Support / Resistance
    df["resistance_20"] = hi.rolling(20).max()
    df["support_20"]    = lo.rolling(20).min()

    return df


# ─── Trend & Pattern Detection ────────────────────────────────────────────────

def detect_trend(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    prev = df.iloc[-20] if len(df) >= 20 else df.iloc[0]

    price_change_pct = (last["close"] - prev["close"]) / (prev["close"] + 1e-9) * 100
    above_sma20 = last["close"] > last.get("sma_20", last["close"])
    above_sma50 = last["close"] > last.get("sma_50", last["close"])
    macd_bull   = last.get("macd", 0) > last.get("macd_signal", 0)
    rsi_val     = last.get("rsi", 50)
    bb_pos      = last.get("bb_pos", 0.5)
    vol_r       = last.get("vol_ratio", 1.0)

    score = 0
    score += 1 if price_change_pct > 0 else -1
    score += 1 if above_sma20 else -1
    score += 1 if above_sma50 else -1
    score += 1 if macd_bull else -1
    score += 1 if rsi_val < 70 else -1
    score += 1 if bb_pos < 0.8 else -1

    atr = last.get("atr", 0)
    volatility_pct = (atr / (last["close"] + 1e-9)) * 100

    recent_5 = df["close"].iloc[-5:].pct_change().sum() * 100

    if score >= 3:
        trend, trend_color = "BULLISH 📈", Fore.GREEN
    elif score <= -3:
        trend, trend_color = "BEARISH 📉", Fore.RED
    else:
        trend, trend_color = "SIDEWAYS / NEUTRAL ↔️", Fore.YELLOW

    vol_label = "HIGH 🔥" if volatility_pct > 3 else "MODERATE" if volatility_pct > 1.5 else "LOW 😴"
    rsi_label = "OVERBOUGHT ⚠️" if rsi_val > 70 else "OVERSOLD 💡" if rsi_val < 30 else "NEUTRAL"

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
        "support": last.get("support_20"),
        "resistance": last.get("resistance_20"),
        "cci": round(last.get("cci", 0), 1),
        "williams_r": round(last.get("williams_r", -50), 1),
        "stoch_k": round(last.get("stoch_k", 50), 1),
        "stoch_d": round(last.get("stoch_d", 50), 1),
        "obv_trend": "Rising" if df["obv"].iloc[-1] > df["obv"].iloc[-5] else "Falling",
        "vol_ratio": round(last.get("vol_ratio", 1), 2),
    }


def detect_anomalies(df: pd.DataFrame) -> list:
    anomalies = []
    if len(df) < 5:
        return anomalies
    df = df.copy()
    df["daily_ret"] = df["close"].pct_change()
    mean_ret = df["daily_ret"].mean()
    std_ret  = df["daily_ret"].std()
    for i, row in df.iterrows():
        ret = row["daily_ret"]
        if pd.isna(ret):
            continue
        z = (ret - mean_ret) / (std_ret + 1e-9)
        if abs(z) > 2.5:
            anomalies.append({
                "date": row["date"].strftime("%Y-%m-%d"),
                "close": row["close"],
                "change_pct": round(ret * 100, 2),
                "z_score": round(z, 2),
                "label": "🔺 SPIKE" if ret > 0 else "🔻 CRASH",
            })
    return anomalies[-10:]


# ─── Prediction (Statistical — backward compatible) ──────────────────────────

def predict_prices(df: pd.DataFrame, n: int = 7) -> list:
    """
    Blended statistical prediction:
    1. Linear regression trend extrapolation (40%)
    2. Holt exponential smoothing (40%)
    3. Momentum carry-forward (20%)
    NEPSE ±10% circuit-breaker enforced per step.
    """
    close = df["close"].dropna().values
    if len(close) < 10:
        raise ValueError("Need at least 10 data points to predict.")

    # 1. Linear regression
    x = np.arange(len(close))
    slope, intercept = np.polyfit(x, close, 1)
    lr_preds = [intercept + slope * (len(close) + i) for i in range(1, n + 1)]

    # 2. Holt's exponential smoothing
    alpha, beta = 0.3, 0.1
    level, trend_es = close[0], 0.0
    for v in close[1:]:
        prev_level = level
        level = alpha * v + (1 - alpha) * (level + trend_es)
        trend_es = beta * (level - prev_level) + (1 - beta) * trend_es
    es_preds = [level + (i + 1) * trend_es for i in range(n)]

    # 3. Momentum
    lookback = min(10, len(close) - 1)
    avg_daily_ret = np.mean(np.diff(close[-lookback:]) / (close[-lookback:-1] + 1e-9))
    mom_preds = [close[-1] * (1 + avg_daily_ret) ** (i + 1) for i in range(n)]

    recent_vol = np.std(np.diff(close[-20:]) / (close[-20:-1] + 1e-9)) if len(close) >= 21 else np.std(np.diff(close) / (close[:-1] + 1e-9))

    predictions = []
    last_date    = df["date"].iloc[-1]
    current_price = close[-1]

    for i in range(n):
        pred_date = last_date + pd.offsets.BDay(i + 1)
        blended   = 0.40 * lr_preds[i] + 0.40 * es_preds[i] + 0.20 * mom_preds[i]

        # Circuit breaker
        blended = float(np.clip(blended, current_price * 0.90, current_price * 1.10))
        band    = blended * recent_vol * 1.5
        prev_p  = predictions[-1]["predicted_close"] if predictions else close[-1]
        daily_change = (blended - prev_p) / (prev_p + 1e-9) * 100

        predictions.append({
            "day":             i + 1,
            "date":            pred_date.strftime("%Y-%m-%d"),
            "predicted_close": round(blended, 2),
            "low_band":        round(blended - band, 2),
            "high_band":       round(blended + band, 2),
            "change_pct":      round(daily_change, 2),
            "direction_prob":  0.5,  # no classifier in standalone mode
            "confidence":      "medium",
        })
        current_price = blended

    return predictions


# ─── Strategy Signals ─────────────────────────────────────────────────────────

def suggest_strategy(trend_info: dict, predictions: list, df: pd.DataFrame) -> list:
    rsi        = trend_info["rsi"]
    score      = trend_info["score"]
    last       = trend_info["last_close"]
    support    = trend_info.get("support")
    resistance = trend_info.get("resistance")
    mom        = trend_info["recent_5d_momentum"]
    cci        = trend_info.get("cci", 0)
    williams_r = trend_info.get("williams_r", -50)
    stoch_k    = trend_info.get("stoch_k", 50)
    pred_5     = predictions[min(4, len(predictions)-1)]["predicted_close"] if predictions else last
    expected_gain = (pred_5 - last) / (last + 1e-9) * 100

    signals = []

    # RSI signals
    if rsi < 30:
        signals.append({"signal": "BUY", "strength": "STRONG", "reason": f"RSI={rsi} deeply oversold (<30). Classic reversal zone.", "emoji": "💚"})
    elif rsi < 45:
        signals.append({"signal": "BUY", "strength": "MODERATE", "reason": f"RSI={rsi} approaching oversold. Accumulation zone.", "emoji": "🟢"})
    elif rsi > 75:
        signals.append({"signal": "SELL", "strength": "STRONG", "reason": f"RSI={rsi} deeply overbought (>75). Correction risk.", "emoji": "🔴"})
    elif rsi > 65:
        signals.append({"signal": "SELL/REDUCE", "strength": "MODERATE", "reason": f"RSI={rsi} approaching overbought. Consider partial profit-taking.", "emoji": "🟡"})

    # CCI signal (extreme readings)
    if cci < -150:
        signals.append({"signal": "BUY", "strength": "TECHNICAL", "reason": f"CCI={cci}: extreme oversold reading, bounce likely.", "emoji": "📊"})
    elif cci > 150:
        signals.append({"signal": "SELL", "strength": "TECHNICAL", "reason": f"CCI={cci}: extreme overbought, mean-reversion risk.", "emoji": "📊"})

    # Williams %R
    if williams_r < -80:
        signals.append({"signal": "BUY", "strength": "TECHNICAL", "reason": f"Williams %R={williams_r}: deep oversold territory.", "emoji": "📈"})
    elif williams_r > -20:
        signals.append({"signal": "SELL", "strength": "TECHNICAL", "reason": f"Williams %R={williams_r}: strong overbought signal.", "emoji": "📉"})

    # Stochastic
    if stoch_k < 20:
        signals.append({"signal": "BUY", "strength": "TECHNICAL", "reason": f"Stochastic %K={stoch_k}: oversold, watch for %K/%D bullish cross.", "emoji": "🎯"})
    elif stoch_k > 80:
        signals.append({"signal": "SELL/REDUCE", "strength": "TECHNICAL", "reason": f"Stochastic %K={stoch_k}: overbought.", "emoji": "⚠️"})

    # Trend score
    if score >= 4:
        signals.append({"signal": "BUY/HOLD", "strength": "STRONG", "reason": "Strong bullish confluence: price above SMAs, positive MACD, upward momentum.", "emoji": "💚"})
    elif score <= -4:
        signals.append({"signal": "SELL/AVOID", "strength": "STRONG", "reason": "Strong bearish confluence: price below SMAs, negative MACD, downward momentum.", "emoji": "🔴"})

    # Model prediction
    if expected_gain > 5:
        signals.append({"signal": "BUY", "strength": "MODEL", "reason": f"Model predicts +{expected_gain:.1f}% gain over 5 sessions.", "emoji": "🤖"})
    elif expected_gain < -5:
        signals.append({"signal": "SELL", "strength": "MODEL", "reason": f"Model predicts {expected_gain:.1f}% decline over 5 sessions.", "emoji": "🤖"})

    # S/R
    if support and last <= support * 1.02:
        signals.append({"signal": "BUY", "strength": "TECHNICAL", "reason": f"Price near 20-day support (NPR {support:.2f}). Good risk/reward.", "emoji": "📊"})
    if resistance and last >= resistance * 0.98:
        signals.append({"signal": "SELL", "strength": "TECHNICAL", "reason": f"Price near 20-day resistance (NPR {resistance:.2f}). Book partial profits.", "emoji": "📊"})

    if abs(mom) < 0.5:
        signals.append({"signal": "HOLD/WAIT", "strength": "CAUTION", "reason": "5-day momentum near zero. Wait for breakout confirmation.", "emoji": "⏳"})

    if not signals:
        signals.append({"signal": "HOLD", "strength": "NEUTRAL", "reason": "No strong signals. Market in balance.", "emoji": "⚖️"})

    return signals


# ─── Display ──────────────────────────────────────────────────────────────────

def print_header(title: str):
    w = 72
    print(); print(Fore.CYAN + "═" * w); print(Fore.CYAN + f"  {title}"); print(Fore.CYAN + "═" * w)

def print_section(title: str):
    print(); print(Fore.YELLOW + f"▶ {title}"); print(Fore.YELLOW + "─" * 60)


def run_analysis(df: pd.DataFrame, symbol: str, n_predict: int):
    print_header(f"NEPSE STOCK ANALYSIS — {symbol.upper()}")

    print_section("Dataset Summary")
    print(f"  Records      : {len(df)}")
    print(f"  Date Range   : {df['date'].iloc[0].strftime('%Y-%m-%d')}  →  {df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  Latest Close : NPR {df['close'].iloc[-1]:,.2f}")
    print(f"  52-Week High : NPR {df['close'].rolling(min(252, len(df))).max().iloc[-1]:,.2f}")
    print(f"  52-Week Low  : NPR {df['close'].rolling(min(252, len(df))).min().iloc[-1]:,.2f}")
    if "volume" in df.columns and df["volume"].sum() > 0:
        print(f"  Avg Vol (20d): {df['volume'].tail(20).mean():,.0f}")

    df = add_indicators(df)
    trend_info = detect_trend(df)

    print_section("Trend & Momentum Analysis")
    tc = trend_info["trend_color"]
    print(f"  Overall Trend    : {tc}{trend_info['trend']}{Style.RESET_ALL}  (score {trend_info['score']:+d}/6)")
    print(f"  20-Day Change    : {trend_info['price_change_pct_20d']:+.2f}%")
    print(f"  5-Day Momentum   : {trend_info['recent_5d_momentum']:+.2f}%")
    print(f"  RSI (14)         : {trend_info['rsi']}  →  {trend_info['rsi_label']}")
    print(f"  CCI (20)         : {trend_info['cci']}")
    print(f"  Williams %R      : {trend_info['williams_r']}")
    print(f"  Stoch %K/%D      : {trend_info['stoch_k']} / {trend_info['stoch_d']}")
    print(f"  MACD             : {'📈 Bullish' if trend_info['macd_bullish'] else '📉 Bearish'}")
    print(f"  Bollinger Pos    : {trend_info['bb_position_pct']}%  (0=lower band, 100=upper band)")
    print(f"  Volatility (ATR) : {trend_info['volatility_pct']}%  →  {trend_info['volatility_label']}")
    print(f"  OBV Trend        : {trend_info['obv_trend']}")
    print(f"  Volume Ratio     : {trend_info['vol_ratio']}x  (vs 20-day avg)")
    if trend_info.get("support"):
        print(f"  Support (20d)    : NPR {trend_info['support']:,.2f}")
    if trend_info.get("resistance"):
        print(f"  Resistance (20d) : NPR {trend_info['resistance']:,.2f}")

    anomalies = detect_anomalies(df)
    print_section(f"Anomaly Detection  ({len(anomalies)} flagged)")
    if anomalies:
        anom_table = [
            [a["date"], f"NPR {a['close']:,.2f}", f"{a['change_pct']:+.2f}%", f"{a['z_score']:+.2f}σ", a["label"]]
            for a in anomalies
        ]
        print(tabulate(anom_table, headers=["Date", "Close", "Change%", "Z-Score", "Type"], tablefmt="simple"))
    else:
        print("  No significant anomalies detected.")

    print_section(f"Price Prediction — Next {n_predict} Trading Days")
    try:
        predictions = predict_prices(df, n_predict)
        pred_table = [
            [p["day"], p["date"], f"NPR {p['predicted_close']:,.2f}", f"NPR {p['low_band']:,.2f}",
             f"NPR {p['high_band']:,.2f}", f"{p['change_pct']:+.2f}%"]
            for p in predictions
        ]
        print(tabulate(pred_table,
                       headers=["Day", "Date", "Predicted", "Low Band", "High Band", "Δ%"],
                       tablefmt="rounded_outline"))
    except ValueError as e:
        print(f"  {Fore.RED}Prediction skipped: {e}")
        predictions = []

    if predictions:
        strategies = suggest_strategy(trend_info, predictions, df)
        print_section("Trading Strategy Signals")
        for s in strategies:
            color = Fore.GREEN if "BUY" in s["signal"] else Fore.RED if "SELL" in s["signal"] else Fore.YELLOW
            print(f"  {s['emoji']}  [{color}{s['signal']}{Style.RESET_ALL}] ({s['strength']})")
            print(f"       {s['reason']}")

    print_section("Risk Summary")
    risks = []
    if trend_info["volatility_label"] == "HIGH 🔥":
        risks.append("🔥 High volatility — size positions carefully, use stop-losses.")
    if trend_info["rsi"] > 70:
        risks.append("⚠️  RSI overbought — correction risk.")
    if trend_info["rsi"] < 30:
        risks.append("⚠️  RSI oversold — may fall further before reversing.")
    if len(anomalies) >= 3:
        risks.append(f"⚡ {len(anomalies)} price anomalies — news-driven spikes/crashes.")
    if abs(trend_info["recent_5d_momentum"]) < 0.5:
        risks.append("💤 Low momentum — consolidation phase, breakout direction unclear.")
    if not risks:
        risks.append("✅ No major risk flags. Maintain regular position review.")
    for r in risks:
        print(f"  {r}")

    print_section("Methodology")
    print("  Statistical blend: Linear Regression (40%) + Holt Smoothing (40%) + Momentum (20%)")
    print("  For ML-powered predictions with sentiment, run: python nepse_live.py --symbol <SYM>")
    print("  ⚠  All predictions are probabilistic. Use stop-losses. This is not financial advice.")

    print(); print(Fore.CYAN + "═" * 72)
    print(Fore.CYAN + "  Analysis complete. Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(Fore.CYAN + "═" * 72); print()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NEPSE Stock Analysis & Prediction Tool v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--old",     help="Path to old/historical data CSV")
    parser.add_argument("--new",     help="Path to recent data CSV")
    parser.add_argument("--file",    help="Path to single combined CSV")
    parser.add_argument("--symbol",  default="STOCK", help="Stock symbol label (e.g. NABIL)")
    parser.add_argument("--predict", type=int, default=7, help="Number of future days to predict (5-10)")
    args = parser.parse_args()

    if args.file:
        df = load_single(args.file)
    elif args.old and args.new:
        df = combine_and_clean(args.old, args.new)
    else:
        print(Fore.RED + "Error: Provide either --file OR both --old and --new.")
        sys.exit(1)

    if len(df) < 10:
        print(Fore.RED + f"Error: Only {len(df)} rows after cleaning. Need at least 10.")
        sys.exit(1)

    run_analysis(df, args.symbol, max(5, min(10, args.predict)))


if __name__ == "__main__":
    main()