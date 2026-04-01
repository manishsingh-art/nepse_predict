#!/usr/bin/env python3
"""
run_all.py — Batch NEPSE Predictor
=====================================
Runs the full ML pipeline for all (or selected) NEPSE symbols in parallel.
Outputs a clean JSON log of predictions + evaluations.

Usage:
    python run_all.py                        # all symbols, max 10 workers
    python run_all.py --symbols NABIL NTC    # specific symbols
    python run_all.py --workers 5            # control concurrency
    python run_all.py --no-ml               # deprecated flag (ignored)
"""

from __future__ import annotations

import os
import sys
import json
import glob
import time
import argparse
import warnings
import contextlib
from datetime import datetime
from pathlib import Path
import concurrent.futures

warnings.filterwarnings("ignore")


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def load_latest_log() -> dict:
    today_file = Path(f"predictions_log-{datetime.now().strftime('%Y-%m-%d')}.json")
    if today_file.exists():
        try:
            return json.loads(today_file.read_text())
        except Exception:
            return {}
    files = sorted(glob.glob("predictions_log-*.json"), reverse=True)
    if files:
        try:
            return json.loads(Path(files[0]).read_text())
        except Exception:
            pass
    return {}


def process_symbol(sym: str, log_data: dict, use_ml: bool = True) -> tuple:
    """Process a single symbol. Returns (symbol, result_dict | None)."""
    try:
        with suppress_output():
            from fetcher import fetch_history
            from pipeline import train_model

            df = fetch_history(sym, years=2)
            if df is None or df.empty or len(df) < 20:
                return sym, None

        today_str = datetime.now().strftime("%Y-%m-%d")
        latest_date = df["date"].iloc[-1].strftime("%Y-%m-%d")
        actual_close = float(df["close"].iloc[-1])

        history = list(log_data.get(sym, []))

        # Evaluate existing predictions
        for entry in history:
            date_val = entry.get("date")
            p_close = entry.get("predicted_close")
            if not date_val or p_close is None:
                continue
            if date_val == latest_date and entry.get("actual_close") is None:
                entry["actual_close"] = round(actual_close, 2)
                entry["error_pct"] = round((actual_close - p_close) / (p_close + 1e-9) * 100, 2)

        # Backfill today if not present
        has_today = any(e.get("date") == today_str for e in history)
        if not has_today and latest_date == today_str and len(df) > 100:
            with suppress_output():
                try:
                    hist_market = fetch_history("NEPSE", years=2) if sym.upper() != "NEPSE" else None
                    hist_model, hist_clean, _, hist_cols = train_model(
                        data=df.iloc[:-1].copy(),
                        symbol=sym,
                        market_data=hist_market,
                        optimise=False,
                        n_folds=min(5, max(3, len(df.iloc[:-1]) // 180)),
                    )
                    bt_pred = hist_model.forecast(hist_clean, hist_cols, horizon=1)[0]
                    bt_p = float(bt_pred.predicted_close)
                    history.append({
                        "date": today_str,
                        "predicted_close": round(bt_p, 2),
                        "actual_close": round(actual_close, 2),
                        "error_pct": round((actual_close - bt_p) / (bt_p + 1e-9) * 100, 2),
                    })
                except Exception:
                    pass

        # Forecast next day
        with suppress_output():
            try:
                if len(df) < 120:
                    return sym, None
                market_df = fetch_history("NEPSE", years=2) if sym.upper() != "NEPSE" else None
                model, clean_df, _, feature_cols = train_model(
                    data=df,
                    symbol=sym,
                    market_data=market_df,
                    optimise=False,
                    n_folds=3,
                )
                next_forecast = model.forecast(clean_df, feature_cols, horizon=1)[0]
                next_p = float(next_forecast.predicted_close)
                next_date = next_forecast.date
            except Exception:
                return sym, None

        # Append if not duplicate
        has_next = any(e.get("date") == next_date for e in history)
        if not has_next:
            history.append({
                "date": next_date,
                "predicted_close": round(next_p, 2),
                "actual_close": None,
                "error_pct": None,
            })

        # Clean + sort
        history = [e for e in history if e.get("date") and e.get("predicted_close") is not None]
        history.sort(key=lambda x: x["date"])

        return sym, history

    except Exception:
        return sym, None


def main():
    ap = argparse.ArgumentParser(description="Batch NEPSE predictor")
    ap.add_argument("--symbols", nargs="+", help="Specific symbols to process")
    ap.add_argument("--workers", type=int, default=8, help="Thread pool size (default 8)")
    ap.add_argument("--no-ml",  action="store_true", help="Deprecated; the batch runner is ML-only now")
    args = ap.parse_args()

    final_output: dict = {}
    t0 = time.time()

    # Load symbols live (cached 24h)
    with suppress_output():
        from fetcher import fetch_nepse_symbols
        syms_map = fetch_nepse_symbols()

    symbols = args.symbols if args.symbols else list(syms_map.keys())
    symbols = [s.upper() for s in symbols if s]
    if not symbols:
        print("{}")
        return

    with suppress_output():
        log_data = load_latest_log()

    use_ml = True

    # Parallel processing
    with suppress_output():
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_symbol, sym, log_data, use_ml): sym
                for sym in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                sym, history = future.result()
                if history:
                    final_output[sym] = history

    # Output JSON
    print(json.dumps(final_output, indent=2))

    # Persist log
    today_file = Path(f"predictions_log-{datetime.now().strftime('%Y-%m-%d')}.json")
    try:
        today_file.write_text(json.dumps(final_output, indent=2))
    except Exception:
        pass

    elapsed = time.time() - t0
    sys.stderr.write(f"run_all.py: processed {len(final_output)} symbols in {elapsed:.1f}s\n")


if __name__ == "__main__":
    main()