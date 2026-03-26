#!/usr/bin/env python3
"""
sector_analysis.py — NEPSE Sector Impact Analysis
==================================================
Identifies market components (sectors) and their influence on the NEPSE Index.
Features:
  - Sector-wise performance tracking.
  - Correlation analysis with the main NEPSE Index.
  - Identification of leading/lagging sectors.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add current directory to path
sys.path.append(os.getcwd())

from fetcher import fetch_company_list, fetch_history

logger = logging.getLogger(__name__)

# Official-ish NEPSE sectors
SECTORS = [
    "Commercial Banks", "Development Banks", "Finance", "Hotels And Tourism",
    "Hydropower", "Investment", "Life Insurance", "Manufacturing And Processing",
    "Microfinance", "Mutual Fund", "Non Life Insurance", "Others", "Trading"
]

def get_sector_mapping() -> Dict[str, str]:
    """Returns a mapping of Symbol -> Sector."""
    try:
        companies = fetch_company_list()
        # Filter out empty sectors if possible, otherwise use built-in fallbacks
        mapping = {}
        for _, row in companies.iterrows():
            sym = row['symbol'].upper()
            sec = row['sector']
            if sec and sec != 'nan' and sec != '':
                mapping[sym] = sec
        
        # Add some manual fallbacks for major stocks if the API was empty
        fallbacks = {
            "NABIL": "Commercial Banks", "NICA": "Commercial Banks", "EBL": "Commercial Banks",
            "NTC": "Telecom", "UPPER": "Hydropower", "NHPC": "Hydropower",
            "NLIC": "Life Insurance", "LICN": "Life Insurance",
            "CBBL": "Microfinance", "SWBBL": "Microfinance",
            "CIT": "Others", "HDL": "Manufacturing And Processing",
            "SHL": "Hotels And Tourism", "TRH": "Hotels And Tourism"
        }
        for sym, sec in fallbacks.items():
            if sym not in mapping or not mapping[sym]:
                mapping[sym] = sec
        return mapping
    except Exception as e:
        logger.error(f"Error getting sector mapping: {e}")
        return {}

def calculate_sector_performance(days: int = 30) -> pd.DataFrame:
    """
    Calculates returns for each sector by averaging returns of top stocks.
    Returns DataFrame[Sector, Return_Pct, Volatility, Correlation_With_NEPSE]
    """
    mapping = get_sector_mapping()
    if not mapping:
        return pd.DataFrame()

    # Get NEPSE Index history as benchmark
    try:
        nepse_df = fetch_history("NEPSE", years=1)
        nepse_df['ret'] = nepse_df['close'].pct_change()
    except Exception:
        nepse_df = pd.DataFrame()

    sector_data = []
    
    # Analyze each sector
    for sector in SECTORS:
        sector_stocks = [sym for sym, sec in mapping.items() if sec == sector]
        if not sector_stocks:
            continue
        
        # Take up to top 5 stocks per sector (simplification)
        # In a production system, we'd use market cap weighting.
        top_stocks = sector_stocks[:5]
        
        stock_rets = []
        for sym in top_stocks:
            try:
                df = fetch_history(sym, years=1)
                df['ret'] = df['close'].pct_change()
                stock_rets.append(df[['date', 'ret']].rename(columns={'ret': sym}))
            except Exception:
                continue
        
        if not stock_rets:
            continue
            
        # Merge all stocks in sector
        merged = stock_rets[0]
        for r in stock_rets[1:]:
            merged = pd.merge(merged, r, on='date', how='inner')
        
        # Average return for the sector
        merged['sector_ret'] = merged[top_stocks].mean(axis=1)
        
        # Calculate metrics
        recent = merged.tail(days)
        if recent.empty:
            continue
            
        avg_ret = recent['sector_ret'].mean() * 100 * days # Approx period return
        vol = recent['sector_ret'].std() * 100
        
        # Correlation with NEPSE
        corr = 0.0
        if not nepse_df.empty:
            # Normalize dates for merging
            merged['date_only'] = pd.to_datetime(merged['date']).dt.date
            nepse_df['date_only'] = pd.to_datetime(nepse_df['date']).dt.date
            
            m_with_nepse = pd.merge(
                merged[['date_only', 'sector_ret']], 
                nepse_df[['date_only', 'ret']], 
                on='date_only'
            )
            if len(m_with_nepse) > 10:
                corr = m_with_nepse['sector_ret'].corr(m_with_nepse['ret'])
        
        sector_data.append({
            "Sector": sector,
            "Return_Period_Pct": round(avg_ret, 2),
            "Volatility": round(vol, 2),
            "Correlation_Index": round(corr, 3),
            "Count": len(top_stocks)
        })
        
    return pd.DataFrame(sector_data).sort_values("Return_Period_Pct", ascending=False)

def identify_market_drivers():
    """Identifies which sectors currently have the most impact."""
    df = calculate_sector_performance(days=60)
    if df.empty:
        print("No sector data available.")
        return
    
    print("\n--- NEPSE Sector Impact Analysis (Last 60 Days) ---")
    # A driver is a sector with high correlation AND high positive return
    df['Impact_Score'] = df['Return_Period_Pct'] * df['Correlation_Index']
    df = df.sort_values("Impact_Score", ascending=False)
    
    from tabulate import tabulate
    print(tabulate(df, headers='keys', tablefmt='rounded_outline', showindex=False))
    
    top_driver = df.iloc[0]['Sector']
    print(f"\n💡 Current Market Leader: {top_driver}")
    print(f"   (Based on high correlation and strong performance)")

if __name__ == "__main__":
    identify_market_drivers()
