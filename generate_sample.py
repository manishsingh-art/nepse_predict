#!/usr/bin/env python3
"""
Generate sample NEPSE-style CSV data for testing analyze.py.
Produces two files: old_data.csv (1 year ago) and new_data.csv (recent 3 months).

Usage:
    python generate_sample.py --symbol NABIL --start-price 1200
"""
import argparse
import numpy as np
import pandas as pd
from datetime import date

def gen_ohlcv(n_days: int, start_price: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=date.today(), periods=n_days)
    prices = [start_price]
    for _ in range(n_days - 1):
        ret = rng.normal(0.001, 0.018)   # slight upward drift, NEPSE-typical vol
        prices.append(round(prices[-1] * (1 + ret), 2))

    rows = []
    for i, (d, cl) in enumerate(zip(dates, prices)):
        op = round(cl * rng.uniform(0.985, 1.015), 2)
        hi = round(max(op, cl) * rng.uniform(1.000, 1.025), 2)
        lo = round(min(op, cl) * rng.uniform(0.975, 1.000), 2)
        vol = int(rng.integers(5_000, 80_000))
        rows.append({"Date": d.strftime("%Y-%m-%d"), "Open": op,
                     "High": hi, "Low": lo, "Close": cl, "Volume": vol})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol",      default="NABIL")
    ap.add_argument("--start-price", type=float, default=1100.0)
    args = ap.parse_args()

    full = gen_ohlcv(300, args.start_price)
    old  = full.iloc[:240]    # first ~240 trading days = ~old historical
    new  = full.iloc[220:]    # last ~80 days = recent (20-day overlap intentional)

    old.to_csv("old_data.csv", index=False)
    new.to_csv("new_data.csv", index=False)
    print(f"✅  old_data.csv  ({len(old)} rows)  &  new_data.csv  ({len(new)} rows)  generated.")
    print(f"    Symbol: {args.symbol}  |  Starting price: NPR {args.start_price:.2f}")
    print()
    print("Run analysis:")
    print(f"  python analyze.py --old old_data.csv --new new_data.csv --symbol {args.symbol} --predict 7")

if __name__ == "__main__":
    main()
