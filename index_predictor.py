#!/usr/bin/env python3
"""
index_predictor.py — Specialized NEPSE Index Predictor
========================================================
Forecasts the overall NEPSE Index using market-wide features
and sector-level performance drivers.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Optional

# Add current directory to path
sys.path.append(os.getcwd())

from fetcher import fetch_history, get_aggregate_sentiment
from features import build_features, add_targets, get_feature_cols
from models import NEPSEEnsemble
from sector_analysis import get_sector_mapping

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def load_sector_impact() -> List[dict]:
    try:
        with open('sector_impact.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def get_top_sector_features(nepse_dates: pd.Series) -> pd.DataFrame:
    """Fetches returns for top correlated sectors as features."""
    impact = load_sector_impact()
    if not impact:
        return pd.DataFrame(index=nepse_dates.index)
    
    # Sort by correlation
    top_sectors = sorted(impact, key=lambda x: x['Correlation_Index'], reverse=True)[:3]
    mapping = get_sector_mapping()
    
    sector_features = pd.DataFrame(index=nepse_dates.index)
    sector_features['date'] = nepse_dates
    
    for sec_info in top_sectors:
        sector = sec_info['Sector']
        # Find a representative stock for this sector
        representative = next((sym for sym, s in mapping.items() if s == sector), None)
        if not representative:
            continue
            
        try:
            logger.info(f"Fetching {sector} driver: {representative}")
            df = fetch_history(representative, years=1)
            df['ret'] = df['close'].pct_change()
            df['date_only'] = pd.to_datetime(df['date']).dt.date
            
            # Merge with index dates
            sector_features['date_only'] = pd.to_datetime(sector_features['date']).dt.date
            temp = pd.merge(sector_features[['date_only']], df[['date_only', 'ret']], on='date_only', how='left')
            sector_features[f"sec_{sector.replace(' ', '_')}_ret"] = temp['ret'].values
        except Exception as e:
            logger.warning(f"Could not fetch features for sector {sector}: {e}")
            
    return sector_features.drop(columns=['date', 'date_only'])

def run_index_prediction(horizon: int = 7):
    print("\n" + "="*60)
    print("      NEPSE INDEX PREDICTION SYSTEM v1.0")
    print("="*60 + "\n")
    
    # 1. Fetch NEPSE Index History
    logger.info("Fetching NEPSE Index history...")
    df = fetch_history("NEPSE", years=5)
    
    # 2. Get News Sentiment for Market
    logger.info("Analyzing market sentiment...")
    sentiment = get_aggregate_sentiment("NEPSE", days_back=7)
    market_sentiment = sentiment['score']
    
    # 3. Add Sector Features
    logger.info("Integrating sector-level drivers...")
    sec_feats = get_top_sector_features(df['date'])
    df = pd.concat([df, sec_feats], axis=1)
    
    # 4. Build Standard Features
    logger.info("Engineering technical indicators...")
    df_feat = build_features(df, sentiment_score=market_sentiment)
    df_feat = add_targets(df_feat)
    
    # Add index-specific features (e.g. cumulative sector impact)
    sec_cols = [c for c in df_feat.columns if c.startswith("sec_") and c.endswith("_ret")]
    if sec_cols:
        df_feat["avg_sector_ret"] = df_feat[sec_cols].mean(axis=1)
    
    feature_cols = get_feature_cols(df_feat)
    if "avg_sector_ret" in df_feat.columns:
        feature_cols.append("avg_sector_ret")
    
    # 5. Train Ensemble
    logger.info(f"Training NEPSE Index Ensemble (rows={len(df_feat)})...")
    ensemble = NEPSEEnsemble(symbol="NEPSE", n_folds=5, optimise=True)
    ensemble.fit(df_feat, feature_cols)
    
    # 6. Forecast
    logger.info(f"Generating {horizon}-day forecast...")
    forecast = ensemble.forecast(df, feature_cols, horizon=horizon, sentiment_score=market_sentiment)
    
    # 7. Output Results
    print("\n--- NEPSE Index Forecast ---")
    from tabulate import tabulate
    table = []
    for p in forecast:
        table.append([p.day, p.date, f"{p.predicted_close:,.2f}", f"{p.change_pct:+.2f}%", p.confidence])
    
    print(tabulate(table, headers=["Day", "Date", "Predicted", "Δ%", "Conf"], tablefmt="rounded_outline"))
    
    # Save Report
    report = {
        "symbol": "NEPSE",
        "last_close": float(df['close'].iloc[-1]),
        "sentiment": sentiment,
        "forecast": [vars(p) for p in forecast],
        "metrics": vars(ensemble.report_) if ensemble.report_ else None
    }
    with open("nepse_index_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✅ Report saved to nepse_index_report.json")

if __name__ == "__main__":
    run_index_prediction()
