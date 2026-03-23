"""
fetcher.py — Live NEPSE Data Fetcher
=====================================
Fetches:
  1. Full list of all NEPSE-listed companies (symbol, name, sector)
  2. Complete OHLCV price history for any symbol

Sources (tried in order):
  Primary   : merolagani.com  (TechnicalChartHandler — free, no auth)
  Secondary : nepalstock.com.np (official NEPSE API — public endpoints)
  Tertiary  : sharesansar.com (HTML scrape fallback)
"""

import os
import time
import json
import math
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".nepse_cache")
COMPANY_CACHE = os.path.join(CACHE_DIR, "companies_cache.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://merolagani.com/",
}

MEROLAGANI_CHART = (
    "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
    "?type=get_advanced_chart"
    "&symbol={symbol}"
    "&resolution=1D"
    "&rangeStartDate={start_ts}"
    "&rangeEndDate={end_ts}"
    "&from=&isAdjust=1&currencyCode=NPR"
)

NEPALSTOCK_COMPANY_LIST = "https://nepalstock.com.np/api/nots/securityDailyTradeReport/58"
NEPALSTOCK_SECURITY_LIST = "https://nepalstock.com.np/api/nots/security?nonDelisted=true"
NEPALSTOCK_HISTORY = (
    "https://nepalstock.com.np/api/nots/market/history/"
    "{id}?startDate={start}&endDate={end}&size=500&page=1"
)

SHARESANSAR_HISTORY = (
    "https://www.sharesansar.com/company/{symbol}"
)

# ─── Helper Functions ─────────────────────────────────────────────────────────

def robust_request(url: str, headers: dict = None, timeout: int = 30,
                  retries: int = 3, backoff: float = 1.0) -> Optional[requests.Response]:
    """Helper to perform requests with retries and exponential backoff."""
    headers = headers or HEADERS
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code in [403, 429]: # Rate limited or blocked
                time.sleep(backoff * (2 ** attempt))
                continue
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                raise e
            time.sleep(backoff * (2 ** attempt))
    return None

# ─── Company List ─────────────────────────────────────────────────────────────

def fetch_company_list() -> pd.DataFrame:
    """
    Return DataFrame with columns: symbol, name, sector, id
    Tries live sources (API + Scraping), then local cache, then built-in.
    """
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)

    all_stocks = {} # Dictionary keyed by symbol to unique-ify

    # ── Try 1: NEPSE Official API (Security List) ───────────────────────────
    endpoints = [
        "https://nepalstock.com.np/api/nots/security?nonDelisted=true",
        "https://nepalstock.com.np/api/nots/company/list"
    ]
    headers = {**HEADERS, "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in endpoints:
        try:
            r = robust_request(url, headers=headers)
            if r is not None and r.status_code == 200:
                data = r.json()
                items = data.get("body", []) if isinstance(data.get("body"), list) else data
                for item in items:
                    sym = item.get("symbol") or item.get("stockSymbol") or item.get("stock_symbol")
                    if sym:
                        sym = sym.strip().upper()
                        name = item.get("companyName") or item.get("stock_name") or item.get("company_name")
                        sec = item.get("businessSector") or {}
                        sector = sec.get("name") if isinstance(sec, dict) else str(sec)
                        sid = item.get("id") or item.get("company_id")
                        all_stocks[sym] = {"symbol": sym, "name": name, "sector": sector, "id": sid}
        except Exception:
            continue

    # ── Try 2: Merolagani Scraping ──────────────────────────────────────────
    try:
        url = "https://merolagani.com/StockQuote.aspx"
        r = robust_request(url, headers=headers)
        if r is not None and r.status_code == 200:
            import re
            symbols = re.findall(r'symbol=([A-Z0-9]{2,8})["\']', r.text)
            for s in symbols:
                s = s.upper()
                if s not in all_stocks:
                    all_stocks[s] = {"symbol": s, "name": "", "sector": "", "id": ""}
    except Exception:
        pass

    # ── Try 3: ShareSansar Scraping ──────────────────────────────────────────
    try:
        url = "https://www.sharesansar.com/today-share-price"
        r = robust_request(url, headers=headers)
        if r is not None and r.status_code == 200:
            tables = pd.read_html(r.text)
            for t in tables:
                if any("symbol" in str(c).lower() or "ticker" in str(c).lower() for c in t.columns):
                    t.columns = [str(c).lower().strip() for c in t.columns]
                    sym_col = next(c for c in t.columns if "symbol" in c or "ticker" in c)
                    for _, row in t.iterrows():
                        sym = str(row[sym_col]).strip().upper()
                        if sym and sym != "NAN" and len(sym) < 8:
                            if sym not in all_stocks:
                                all_stocks[sym] = {"symbol": sym, "name": "", "sector": "", "id": ""}
    except Exception:
        pass

    # If we found enough stocks, update cache and return
    if len(all_stocks) > 50:
        df = pd.DataFrame(list(all_stocks.values())).sort_values("symbol").reset_index(drop=True)
        df.to_csv(COMPANY_CACHE, index=False)
        print(f"  ✅ Discovery complete: {len(df)} stocks identified and cached.")
        return df

    # ── Try 4: Local Cache (Persistent Discovery Fallback) ───────────────────
    if os.path.exists(COMPANY_CACHE):
        try:
            df = pd.read_csv(COMPANY_CACHE)
            # Ensure symbols are upper case
            df["symbol"] = df["symbol"].astype(str).str.upper()
            if len(df) > 50:
                print(f"  ✅ Loaded {len(df)} stocks from local cache (offline mode).")
                return df
        except Exception:
            pass

    # ── Fallback: built-in list (Absolute Last Resort) ──────────────────────
    print("  ⚠ Automatic discovery failed. Using built-in symbol list fallback.")
    return _builtin_company_list()


def _builtin_company_list() -> pd.DataFrame:
    """Built-in curated NEPSE symbol list as absolute fallback."""
    stocks = [
        # Banking
        ("NABIL", "Nabil Bank Ltd", "Commercial Banks"),
        ("NICA",  "NIC Asia Bank Ltd", "Commercial Banks"),
        ("SCB",   "Standard Chartered Bank Nepal", "Commercial Banks"),
        ("SANIMA","Sanima Bank Ltd", "Commercial Banks"),
        ("MBL",   "Machhapuchchhre Bank Ltd", "Commercial Banks"),
        ("PRVU",  "Prabhu Bank Ltd", "Commercial Banks"),
        ("NBL",   "Nepal Bank Ltd", "Commercial Banks"),
        ("RBB",   "Rastriya Banijya Bank Ltd", "Commercial Banks"),
        ("EBL",   "Everest Bank Ltd", "Commercial Banks"),
        ("HBL",   "Himalayan Bank Ltd", "Commercial Banks"),
        ("KBL",   "Kumari Bank Ltd", "Commercial Banks"),
        ("LBL",   "Laxmi Sunrise Bank Ltd", "Commercial Banks"),
        ("NIMB",  "Nepal Investment Mega Bank Ltd", "Commercial Banks"),
        ("ADBL",  "Agricultural Development Bank Ltd", "Commercial Banks"),
        ("SBI",   "Nepal SBI Bank Ltd", "Commercial Banks"),
        ("SRBL",  "Sunrise Bank Ltd", "Commercial Banks"),
        ("GBIME", "Global IME Bank Ltd", "Commercial Banks"),
        ("CBL",   "Civil Bank Ltd", "Commercial Banks"),
        ("PCBL",  "Prime Commercial Bank Ltd", "Commercial Banks"),
        ("NMB",   "NMB Bank Ltd", "Commercial Banks"),
        ("CZBIL", "Citizens Bank International Ltd", "Commercial Banks"),
        ("BOKL",  "Bank of Kathmandu Ltd", "Commercial Banks"),
        ("MEGA",  "Mega Bank Nepal Ltd", "Commercial Banks"),
        # Development Banks
        ("MLBSL", "Muktinath Bikas Bank Ltd", "Development Banks"),
        ("MNBBL", "Manakamana Nepal Bikas Bank Ltd", "Development Banks"),
        ("SHINE", "Shine Resunga Development Bank", "Development Banks"),
        # Finance
        ("SIFC",  "Siddhartha Insurance Finance Company", "Finance"),
        ("GUFL",  "Guheshwori Merchant Banking", "Finance"),
        # Insurance
        ("LICN",  "Life Insurance Corporation Nepal", "Life Insurance"),
        ("NLIC",  "National Life Insurance", "Life Insurance"),
        ("ALICL", "Asian Life Insurance Co. Ltd", "Life Insurance"),
        ("PLIC",  "Prime Life Insurance", "Life Insurance"),
        ("SLICL", "Surya Life Insurance", "Life Insurance"),
        ("SICL",  "Siddhartha Insurance Ltd", "Non-Life Insurance"),
        ("NIL",   "Nepal Insurance Ltd", "Non-Life Insurance"),
        ("PIC",   "Premier Insurance Co.", "Non-Life Insurance"),
        ("SIC",   "Sagarmatha Insurance Co.", "Non-Life Insurance"),
        ("NBI",   "Nepal Bangladesh Insurance", "Non-Life Insurance"),
        # Hydropower
        ("CHCL",  "Chilime Hydropower Co.", "Hydropower"),
        ("NHDL",  "Nepal Hydro Developers Ltd", "Hydropower"),
        ("BPCL",  "Butwal Power Company Ltd", "Hydropower"),
        ("RURU",  "Rural Microfinance Dev. Centre", "Hydropower"),
        ("AKPL",  "Arun Kabeli Power Ltd", "Hydropower"),
        ("UPPER", "Upper Tamakoshi Hydropower", "Hydropower"),
        ("NHPC",  "National Hydropower Company", "Hydropower"),
        ("KPCL",  "Khani Khola Hydropower Co.", "Hydropower"),
        ("SHPC",  "Sanima Mai Hydropower", "Hydropower"),
        ("API",   "Api Power Company Ltd", "Hydropower"),
        ("RRHP",  "Ridi Hydropower Development", "Hydropower"),
        # Microfinance
        ("CBBL",  "Chhimek Bikas Bank Ltd", "Microfinance"),
        ("SWBBL", "Swabalamban Bikas Bank Ltd", "Microfinance"),
        ("SKBBL", "Sudur Pashchim Bikas Bank Ltd", "Microfinance"),
        ("SMFDB", "Sana Kisan Bikas Bank", "Microfinance"),
        ("DDBL",  "Deprosc Development Bank", "Microfinance"),
        # Others
        ("NTC",   "Nepal Telecom", "Telecom"),
        ("NIFRA", "Nepal Infrastructure Bank", "Infrastructure"),
        ("BNL",   "Bottlers Nepal Ltd", "Manufacturing"),
        ("UNL",   "Unilever Nepal Ltd", "Manufacturing"),
        ("SHIVM", "Shivam Cement Ltd", "Manufacturing"),
        ("CIT",   "Citizen Investment Trust", "Others"),
        ("NMBMF", "NMB Microfinance", "Microfinance"),
        ("MSMBS", "Mahuli Samudayik Laghubitta", "Microfinance"),
    ]
    return pd.DataFrame(stocks, columns=["symbol", "name", "sector"]).assign(id="")


# ─── Historical Price Data ────────────────────────────────────────────────────

def fetch_history(symbol: str, company_id: str = "",
                  years: int = 3) -> pd.DataFrame:
    """
    Return OHLCV DataFrame for `symbol` covering `years` years.
    Tries merolagani first, then nepalstock, then sharesansar.
    """
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years)

    df = _fetch_merolagani(symbol, start_dt, end_dt)
    if df is not None and len(df) >= 10:
        return df

    # If ID is missing, try to find it in company list or scrape it
    if not company_id:
        try:
            clist = fetch_company_list()
            match = clist[clist["symbol"].str.upper() == symbol.upper()]
            if not match.empty:
                company_id = str(match.iloc[0].get("id", ""))
        except Exception:
            pass

    if company_id:
        df = _fetch_nepalstock(company_id, start_dt, end_dt)
        if df is not None and len(df) >= 10:
            return df

    df = _fetch_sharesansar(symbol)
    if df is not None and len(df) >= 10:
        return df

    raise RuntimeError(
        f"Could not fetch data for '{symbol}' from any source.\n"
        "Please check the symbol is correct and you have internet access."
    )


def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise and clean an OHLCV dataframe."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


def _fetch_merolagani(symbol: str, start: datetime,
                      end: datetime) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from merolagani TechnicalChartHandler."""
    try:
        start_ts = int(start.timestamp())
        end_ts   = int(end.timestamp())
        url = MEROLAGANI_CHART.format(
            symbol=symbol.upper(),
            start_ts=start_ts,
            end_ts=end_ts
        )
        r = robust_request(url, headers={
            **HEADERS,
            "Referer": f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}"
        })

        if r is None or r.status_code != 200:
            return None

        data = r.json()
        # API returns: {"t": [timestamps], "o": [open], "h": [high],
        #               "l": [low], "c": [close], "v": [volume], "s": "ok"}
        if data.get("s") != "ok" or not data.get("t"):
            return None

        df = pd.DataFrame({
            "date":   [datetime.fromtimestamp(ts) for ts in data["t"]],
            "open":   data.get("o", [None] * len(data["t"])),
            "high":   data.get("h", [None] * len(data["t"])),
            "low":    data.get("l", [None] * len(data["t"])),
            "close":  data["c"],
            "volume": data.get("v", [0] * len(data["t"])),
        })
        df = _to_ohlcv(df)
        print(f"  ✅ merolagani.com  →  {len(df)} rows  ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
        return df

    except Exception as e:
        print(f"  ⚠ merolagani fetch error for {symbol}: {e}")
        return None


def _fetch_nepalstock(company_id: str, start: datetime,
                      end: datetime) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from nepalstock.com.np official API."""
    try:
        url = NEPALSTOCK_HISTORY.format(
            id=company_id,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        r = robust_request(url)
        if r is None or r.status_code != 200:
            return None
        data = r.json()
        items = data.get("body", {}).get("data", [])
        if not items:
            return None
        rows = []
        for item in items:
            rows.append({
                "date":   item.get("businessDate"),
                "open":   item.get("openPrice"),
                "high":   item.get("highPrice"),
                "low":    item.get("lowPrice"),
                "close":  item.get("closingPrice") or item.get("lastTradedPrice"),
                "volume": item.get("totalTradeQuantity"),
            })
        df = _to_ohlcv(pd.DataFrame(rows))
        if len(df) > 0:
            print(f"  ✅ nepalstock.com.np  →  {len(df)} rows  ({df['date'].iloc[0].date()} to {df['date'].iloc[-1].date()})")
        return df if len(df) > 0 else None
    except Exception as e:
        print(f"  ⚠ nepalstock fetch error: {e}")
        return None


def _fetch_sharesansar(symbol: str) -> Optional[pd.DataFrame]:
    """Scrape ShareSansar price history table as last resort."""
    try:
        import re
        sess = requests.Session()
        sess.headers.update(HEADERS)
        r = robust_request(
            f"https://www.sharesansar.com/company/{symbol.lower()}"
        )
        if r is None or r.status_code != 200:
            return None
        
        tables = pd.read_html(r.text)
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("close" in c or "ltp" in c for c in cols):
                t.columns = [str(c).lower().strip() for c in t.columns]
                # normalise column names
                rename = {}
                for c in t.columns:
                    if "date" in c:   rename[c] = "date"
                    elif "open" in c: rename[c] = "open"
                    elif "high" in c: rename[c] = "high"
                    elif "low" in c:  rename[c] = "low"
                    elif "close" in c or "ltp" in c: rename[c] = "close"
                    elif "vol" in c or "qty" in c:   rename[c] = "volume"
                t = t.rename(columns=rename)
                if "close" in t.columns:
                    df = _to_ohlcv(t)
                    if len(df) > 0:
                        print(f"  ✅ sharesansar.com  →  {len(df)} rows")
                        return df
    except Exception as e:
        print(f"  ⚠ sharesansar fetch error for {symbol}: {e}")
    return None
