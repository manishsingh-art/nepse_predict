#!/usr/bin/env python3
"""
fetcher.py — Production NEPSE Data Fetcher
===========================================
Multi-source OHLCV + sentiment data pipeline.

Sources (tried in priority order):
  1. merolagani.com  (TechnicalChartHandler — free, fast)
  2. nepalstock.com.np (official NEPSE REST API)
  3. sharesansar.com  (HTML scrape fallback)

Sentiment sources:
  - NewsAPI (free tier: 100 req/day)
  - RSS feeds: The Himalayan Times, Republica, OnlineKhabar
  - Reddit r/investing scrape (public JSON API)
"""

from __future__ import annotations

import os
import re
import time
import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Cache ──────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(os.path.expanduser("~")) / ".nepse_cache"
COMPANY_CACHE = CACHE_DIR / "companies_cache.csv"
PRICE_CACHE_DIR = CACHE_DIR / "price_cache"
NEWS_CACHE_DIR = CACHE_DIR / "news_cache"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://merolagani.com/",
}

MEROLAGANI_CHART = (
    "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
    "?type=get_advanced_chart&symbol={symbol}&resolution=1D"
    "&rangeStartDate={start_ts}&rangeEndDate={end_ts}&from=&isAdjust=1&currencyCode=NPR"
)

NEPALSTOCK_SECURITY_LIST = "https://nepalstock.com.np/api/nots/security?nonDelisted=true"
NEPALSTOCK_HISTORY = (
    "https://nepalstock.com.np/api/nots/market/history/"
    "{id}?startDate={start}&endDate={end}&size=500&page=1"
)

COL_ALIASES = {
    "date":   ["date", "Date", "DATE", "trading_date", "Trading Date"],
    "open":   ["open", "Open", "OPEN", "open_price", "Open Price"],
    "high":   ["high", "High", "HIGH", "high_price", "High Price"],
    "low":    ["low", "Low", "LOW", "low_price", "Low Price"],
    "close":  ["close", "Close", "CLOSE", "close_price", "Close Price", "ltp", "LTP", "last"],
    "volume": ["volume", "Volume", "VOLUME", "qty", "Quantity", "turnover", "Turnover"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    for d in [CACHE_DIR, PRICE_CACHE_DIR, NEWS_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def robust_request(
    url: str,
    headers: dict = None,
    timeout: int = 30,
    retries: int = 3,
    backoff: float = 1.5,
    method: str = "GET",
    json_body: dict = None,
    verify: bool = True,
) -> Optional[requests.Response]:
    headers = headers or HEADERS
    for attempt in range(retries):
        try:
            kwargs = {"headers": headers, "timeout": timeout, "verify": verify}
            if method == "POST":
                r = requests.post(url, json=json_body, **kwargs)
            else:
                r = requests.get(url, **kwargs)
            
            if r.status_code == 200:
                return r
            if r.status_code in (403, 429, 503):
                time.sleep(backoff * (2 ** attempt))
            else:
                return r  # return non-200 for caller to inspect
        except requests.exceptions.RequestException as exc:
            if attempt == retries - 1:
                logger.debug("Request failed after %d retries: %s — %s", retries, url, exc)
                return None
            time.sleep(backoff * (2 ** attempt))
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for std, aliases in COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = std
                break
    df = df.rename(columns=col_map)
    if "close" not in df.columns:
        raise ValueError(f"Cannot find 'close' column. Found: {list(df.columns)}")
    if "date" not in df.columns:
        raise ValueError(f"Cannot find 'date' column. Found: {list(df.columns)}")
    return df


def _to_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
            )
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"].astype(str).str.replace(",", "").str.strip(), errors="coerce"
        )
    else:
        df["volume"] = np.nan
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return df


# ── Company List ──────────────────────────────────────────────────────────────

def fetch_company_list() -> pd.DataFrame:
    """Return DataFrame[symbol, name, sector, id]. Caches to disk."""
    _ensure_dirs()
    all_stocks: Dict[str, dict] = {}

    # Source 1: NEPSE official API
    for url in [NEPALSTOCK_SECURITY_LIST, "https://nepalstock.com.np/api/nots/company/list"]:
        try:
            r = robust_request(url)
            if r is None:
                continue
            data = r.json()
            items = data.get("body", data) if isinstance(data.get("body"), list) else data
            if not isinstance(items, list):
                continue
            for item in items:
                sym = (item.get("symbol") or item.get("stockSymbol") or "").strip().upper()
                if not sym:
                    continue
                sec = item.get("businessSector") or {}
                all_stocks[sym] = {
                    "symbol": sym,
                    "name": item.get("companyName") or item.get("stock_name") or sym,
                    "sector": sec.get("name", "") if isinstance(sec, dict) else str(sec),
                    "id": str(item.get("id") or item.get("company_id") or ""),
                }
            if len(all_stocks) > 50:
                break
        except Exception:
            continue

    # Source 2: MeroLagani scrape
    if len(all_stocks) < 30:
        try:
            r = robust_request("https://merolagani.com/StockQuote.aspx")
            if r:
                for sym in re.findall(r'symbol=([A-Z0-9]{2,8})["\']', r.text):
                    if sym not in all_stocks:
                        all_stocks[sym] = {"symbol": sym, "name": "", "sector": "", "id": ""}
        except Exception:
            pass

    # Source 3: ShareSansar
    if len(all_stocks) < 30:
        try:
            r = robust_request("https://www.sharesansar.com/today-share-price")
            if r:
                tables = pd.read_html(r.text)
                for t in tables:
                    cols = [str(c).lower() for c in t.columns]
                    if any("symbol" in c or "ticker" in c for c in cols):
                        t.columns = cols
                        sym_col = next(c for c in cols if "symbol" in c or "ticker" in c)
                        for sym in t[sym_col].astype(str).str.upper():
                            if len(sym) <= 8 and sym != "NAN" and sym not in all_stocks:
                                all_stocks[sym] = {"symbol": sym, "name": "", "sector": "", "id": ""}
        except Exception:
            pass

    if len(all_stocks) > 50:
        df = pd.DataFrame(list(all_stocks.values())).sort_values("symbol").reset_index(drop=True)
        df.to_csv(COMPANY_CACHE, index=False)
        return df

    # Local cache fallback
    if COMPANY_CACHE.exists():
        try:
            df = pd.read_csv(COMPANY_CACHE)
            df["symbol"] = df["symbol"].astype(str).str.upper()
            if len(df) > 20:
                return df
        except Exception:
            pass

    return _builtin_company_list()


def _builtin_company_list() -> pd.DataFrame:
    stocks = [
        ("NABIL", "Nabil Bank Ltd", "Commercial Banks"),
        ("NICA", "NIC Asia Bank Ltd", "Commercial Banks"),
        ("SCB", "Standard Chartered Bank Nepal", "Commercial Banks"),
        ("SANIMA", "Sanima Bank Ltd", "Commercial Banks"),
        ("MBL", "Machhapuchchhre Bank Ltd", "Commercial Banks"),
        ("PRVU", "Prabhu Bank Ltd", "Commercial Banks"),
        ("NBL", "Nepal Bank Ltd", "Commercial Banks"),
        ("EBL", "Everest Bank Ltd", "Commercial Banks"),
        ("HBL", "Himalayan Bank Ltd", "Commercial Banks"),
        ("KBL", "Kumari Bank Ltd", "Commercial Banks"),
        ("NIMB", "Nepal Investment Mega Bank Ltd", "Commercial Banks"),
        ("ADBL", "Agricultural Development Bank Ltd", "Commercial Banks"),
        ("GBIME", "Global IME Bank Ltd", "Commercial Banks"),
        ("NMB", "NMB Bank Ltd", "Commercial Banks"),
        ("PCBL", "Prime Commercial Bank Ltd", "Commercial Banks"),
        ("UPPER", "Upper Tamakoshi Hydropower", "Hydropower"),
        ("NHPC", "National Hydropower Company", "Hydropower"),
        ("CHCL", "Chilime Hydropower Co.", "Hydropower"),
        ("BPCL", "Butwal Power Company Ltd", "Hydropower"),
        ("NTC", "Nepal Telecom", "Telecom"),
        ("NLIC", "National Life Insurance", "Life Insurance"),
        ("LICN", "Life Insurance Corporation Nepal", "Life Insurance"),
        ("CBBL", "Chhimek Bikas Bank Ltd", "Microfinance"),
        ("SWBBL", "Swabalamban Bikas Bank Ltd", "Microfinance"),
        ("CIT", "Citizen Investment Trust", "Others"),
        ("UNL", "Unilever Nepal Ltd", "Manufacturing"),
    ]
    return pd.DataFrame(stocks, columns=["symbol", "name", "sector"]).assign(id="")


# ── Historical Price Data ─────────────────────────────────────────────────────

def fetch_history(symbol: str, company_id: str = "", years: int = 5) -> pd.DataFrame:
    """
    Return cleaned OHLCV DataFrame for `symbol` covering `years` years.
    Tries sources in order: MeroLagani → NepalStock → ShareSansar → cache.
    """
    _ensure_dirs()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years)

    # Check disk cache (fresh within 1 hour)
    cache_file = PRICE_CACHE_DIR / f"{symbol.upper()}_{years}y.pkl"
    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 3600:
            try:
                df = pd.read_pickle(cache_file)
                logger.debug("Price cache hit for %s", symbol)
                return df
            except Exception:
                pass

    df = _fetch_merolagani(symbol, start_dt, end_dt)
    if df is not None and len(df) >= 20:
        df.to_pickle(cache_file)
        return df

    if not company_id:
        try:
            companies = fetch_company_list()
            m = companies[companies["symbol"].str.upper() == symbol.upper()]
            if not m.empty:
                company_id = str(m.iloc[0].get("id", ""))
        except Exception:
            pass

    if company_id:
        df = _fetch_nepalstock(company_id, start_dt, end_dt)
        if df is not None and len(df) >= 20:
            df.to_pickle(cache_file)
            return df

    df = _fetch_sharesansar(symbol)
    if df is not None and len(df) >= 20:
        df.to_pickle(cache_file)
        return df

    # Try loading stale cache as last resort
    if cache_file.exists():
        try:
            df = pd.read_pickle(cache_file)
            if len(df) >= 20:
                logger.warning("Using stale cache for %s", symbol)
                return df
        except Exception:
            pass

    raise RuntimeError(
        f"Could not fetch data for '{symbol}' from any source. "
        "Check the symbol and internet connection."
    )


def _fetch_merolagani(symbol: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    try:
        url = MEROLAGANI_CHART.format(
            symbol=symbol.upper(),
            start_ts=int(start.timestamp()),
            end_ts=int(end.timestamp()),
        )
        r = robust_request(url, headers={
            **HEADERS,
            "Referer": f"https://merolagani.com/CompanyDetail.aspx?symbol={symbol}",
        })
        if r is None:
            return None
        data = r.json()
        if data.get("s") != "ok" or not data.get("t"):
            return None
        df = pd.DataFrame({
            "date": [datetime.fromtimestamp(ts, tz=timezone.utc).date() for ts in data["t"]],
            "open": data.get("o", [np.nan] * len(data["t"])),
            "high": data.get("h", [np.nan] * len(data["t"])),
            "low": data.get("l", [np.nan] * len(data["t"])),
            "close": data["c"],
            "volume": data.get("v", [0] * len(data["t"])),
        })
        return _to_ohlcv(df)
    except Exception as exc:
        logger.debug("MeroLagani fetch error for %s: %s", symbol, exc)
        return None


def _fetch_nepalstock(company_id: str, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    try:
        url = NEPALSTOCK_HISTORY.format(
            id=company_id,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
        )
        r = robust_request(url)
        if r is None:
            return None
        data = r.json()
        items = data.get("body", {}).get("data", [])
        if not items:
            return None
        rows = [
            {
                "date": item.get("businessDate"),
                "open": item.get("openPrice"),
                "high": item.get("highPrice"),
                "low": item.get("lowPrice"),
                "close": item.get("closingPrice") or item.get("lastTradedPrice"),
                "volume": item.get("totalTradeQuantity"),
            }
            for item in items
        ]
        return _to_ohlcv(pd.DataFrame(rows))
    except Exception as exc:
        logger.debug("NepalStock fetch error: %s", exc)
        return None


def fetch_live_price(symbol: str) -> Optional[float]:
    """Scrape the latest LTP for a symbol using multiple sources (real-time)."""
    sym = symbol.upper().strip()
    
    # Source 1: NEPSE Official Live JSON
    try:
        url = "https://nepalstock.com.np/api/nots/market/active-securities"
        r = robust_request(url, timeout=5, verify=False)
        if r:
            data = r.json()
            for item in data:
                if item.get("symbol") == sym:
                    price = item.get("lastTradedPrice") or item.get("closePrice")
                    if price and float(price) > 0: return float(price)
    except Exception:
        pass

    # Source 2: MeroLagani Today's Share Price
    try:
        r = robust_request("https://merolagani.com/LatestMarket.aspx", timeout=8, verify=False)
        if r:
            tables = pd.read_html(r.text)
            for t in tables:
                cols = [str(c).upper() for c in t.columns]
                if "SYMBOL" in cols:
                    sym_idx = cols.index("SYMBOL")
                    row = t[t.iloc[:, sym_idx].astype(str).str.upper().str.strip() == sym]
                    if not row.empty:
                        # Find LTP column - specifically look for "LTP" text
                        ltp_idx = next((i for i, c in enumerate(cols) if "LTP" in c), 1)
                        val = row.iloc[0, ltp_idx]
                        price = float(str(val).replace(",", ""))
                        if price > 0: return price
    except Exception:
        pass

    # Source 3: ShareSansar Today Share Price
    try:
        r = robust_request("https://www.sharesansar.com/today-share-price", timeout=8, verify=False)
        if r:
            tables = pd.read_html(r.text)
            for t in tables:
                cols = [str(c).lower() for c in t.columns]
                if "symbol" in cols:
                    sym_idx = cols.index("symbol")
                    row = t[t.iloc[:, sym_idx].astype(str).str.upper().str.strip() == sym]
                    if not row.empty:
                        # ShareSansar: LTP is usually index cols.index("ltp")
                        ltp_idx = cols.index("ltp") if "ltp" in cols else 1
                        val = row.iloc[0, ltp_idx]
                        price = float(str(val).replace(",", ""))
                        if price > 0: return price
    except Exception:
        pass

    return None


def _fetch_sharesansar(symbol: str) -> Optional[pd.DataFrame]:
    try:
        r = robust_request(f"https://www.sharesansar.com/company/{symbol.lower()}")
        if r is None: return None
        tables = pd.read_html(r.text)
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any("close" in c or "ltp" in c for c in cols):
                rename = {o: "date" for o in t.columns if "date" in str(o).lower()}
                rename.update({o: "open" for o in t.columns if "open" in str(o).lower()})
                rename.update({o: "high" for o in t.columns if "high" in str(o).lower()})
                rename.update({o: "low" for o in t.columns if "low" in str(o).lower()})
                rename.update({o: "close" for o in t.columns if "close" in str(o).lower() or "ltp" in str(o).lower()})
                rename.update({o: "volume" for o in t.columns if "vol" in str(o).lower() or "qty" in str(o).lower()})
                t = t.rename(columns=rename)
                if "close" in t.columns:
                    df = _to_ohlcv(t)
                    if len(df) > 0: return df
    except Exception:
        pass
    return None
    

# ── Floorsheet & Smart Money Data ─────────────────────────────────────────────

def fetch_floorsheet(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch the latest floorsheet (intraday transactions) for a symbol 
    from ShareSansar. Used to detect buyer/seller concentration.
    """
    symbol = symbol.upper()
    try:
        url = f"https://www.sharesansar.com/floorsheet?symbol={symbol}"
        r = robust_request(url, headers={**HEADERS, "Referer": "https://www.sharesansar.com/"})
        if r is None:
            return None
        
        tables = pd.read_html(r.text)
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if "buyer" in cols and "seller" in cols:
                # Rename columns for internal consistency
                t = t.rename(columns={
                    c: "buyer_broker" for c in t.columns if "buyer" in c.lower()
                })
                t = t.rename(columns={
                    c: "seller_broker" for c in t.columns if "seller" in c.lower()
                })
                t = t.rename(columns={
                    c: "quantity" for c in t.columns if "qty" in c.lower() or "quantity" in c.lower()
                })
                t = t.rename(columns={
                    c: "rate" for c in t.columns if "rate" in c.lower() or "price" in c.lower()
                })
                return t
    except Exception as e:
        logger.debug("Floorsheet fetch error for %s: %s", symbol, e)
    return None


def get_smart_money_signals(symbol: str) -> Dict[str, Any]:
    """
    Analyzes floorsheet for broker-wise concentration.
    Returns: {
        'buy_concentration_top5': float,
        'sell_concentration_top5': float,
        'net_broker_flow': float,
        'smart_money_sentiment': str
    }
    """
    df = fetch_floorsheet(symbol)
    if df is None or len(df) == 0:
        return {
            "buy_concentration_top5": 0.0,
            "sell_concentration_top5": 0.0,
            "net_broker_flow": 0.0,
            "smart_money_sentiment": "NEUTRAL"
        }
    
    try:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        total_vol = df["quantity"].sum()
        
        # Group by brokers
        buyers = df.groupby("buyer_broker")["quantity"].sum().sort_values(ascending=False)
        sellers = df.groupby("seller_broker")["quantity"].sum().sort_values(ascending=False)
        
        top5_buy = buyers.head(5).sum()
        top5_sell = sellers.head(5).sum()
        
        buy_conc = (top5_buy / total_vol) if total_vol > 0 else 0
        sell_conc = (top5_sell / total_vol) if total_vol > 0 else 0
        
        net_flow = top5_buy - top5_sell
        
        # Heuristic: If top 5 buyers occupy > 40% of volume and net flow is positive, Sm Money is Accumulating.
        sentiment = "NEUTRAL"
        if buy_conc > 0.4 and net_flow > 0:
            sentiment = "ACCUMULATION 🏦"
        elif sell_conc > 0.4 and net_flow < 0:
            sentiment = "DISTRIBUTION 📉"
        elif abs(buy_conc - sell_conc) < 0.05:
            sentiment = "CHOPPY / RETAIL 🤝"
            
        return {
            "buy_concentration_top5": round(float(buy_conc), 3),
            "sell_concentration_top5": round(float(sell_conc), 3),
            "net_broker_flow": float(net_flow),
            "smart_money_sentiment": sentiment,
            "top_buyer": int(buyers.index[0]) if not buyers.empty else None,
            "top_seller": int(sellers.index[0]) if not sellers.empty else None
        }
    except Exception:
        return {
            "buy_concentration_top5": 0.0,
            "sell_concentration_top5": 0.0,
            "net_broker_flow": 0.0,
            "smart_money_sentiment": "ERROR"
        }


# ── Sentiment / News Data ─────────────────────────────────────────────────────

RSS_FEEDS = [
    # General Nepal finance/business news
    "https://thehimalayantimes.com/feed/",
    "https://myrepublica.nagariknetwork.com/feed/",
    "https://english.onlinekhabar.com/feed",
    "https://kathmandupost.com/rss",
    "https://sharesansar.com/rss/news",
    "https://merolagani.com/rss/news",
]


def fetch_news_sentiment(symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
    """
    Fetch news articles mentioning `symbol` from RSS feeds.
    Returns list of dicts: {title, published, source, raw_score, label}
    where raw_score is basic keyword polarity (-1..+1).
    """
    _ensure_dirs()
    cache_key = hashlib.md5(f"{symbol}_{days_back}".encode()).hexdigest()[:12]
    cache_file = NEWS_CACHE_DIR / f"{cache_key}.json"

    # Fresh cache within 2 hours
    if cache_file.exists() and (time.time() - cache_file.stat().st_mtime) < 7200:
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception:
            pass

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    for feed_url in RSS_FEEDS:
        try:
            r = robust_request(feed_url, timeout=10, retries=2)
            if r is None:
                continue
            _parse_rss_feed(r.text, symbol, cutoff, articles)
        except Exception:
            continue

    # Deduplicate by title
    seen_titles: set = set()
    unique_articles = []
    for art in articles:
        key = art["title"][:60].lower()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_articles.append(art)

    # Score sentiment
    for art in unique_articles:
        score, label = _keyword_sentiment(art["title"] + " " + art.get("summary", ""))
        art["raw_score"] = score
        art["label"] = label

    # Sort by date desc
    unique_articles.sort(key=lambda x: x.get("published", ""), reverse=True)

    try:
        with open(cache_file, "w") as f:
            json.dump(unique_articles, f, indent=2, default=str)
    except Exception:
        pass

    return unique_articles


def _parse_rss_feed(xml_text: str, symbol: str, cutoff: datetime, out: List) -> None:
    """Parse RSS XML text and append relevant articles to `out`."""
    try:
        # Very lightweight XML parse — avoids lxml dependency issues
        items = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)
        for item in items:
            title = re.search(r"<title[^>]*>(.*?)</title>", item, re.DOTALL)
            link = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
            pub = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
            desc = re.search(r"<description[^>]*>(.*?)</description>", item, re.DOTALL)

            title_text = _strip_cdata(title.group(1)) if title else ""
            desc_text = _strip_cdata(desc.group(1)) if desc else ""

            # Relevance filter: symbol OR generic market keywords
            combined = (title_text + " " + desc_text).upper()
            if not (symbol.upper() in combined or _is_market_relevant(combined)):
                continue

            pub_str = pub.group(1).strip() if pub else ""
            pub_dt = _parse_rss_date(pub_str)
            if pub_dt and pub_dt < cutoff:
                continue

            out.append({
                "title": title_text.strip(),
                "link": (link.group(1).strip() if link else ""),
                "published": pub_str,
                "published_dt": pub_dt.isoformat() if pub_dt else "",
                "summary": re.sub(r"<[^>]+>", " ", desc_text)[:300].strip(),
            })
    except Exception:
        pass


def _strip_cdata(s: str) -> str:
    s = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", s, flags=re.DOTALL)
    return re.sub(r"<[^>]+>", " ", s).strip()


def _parse_rss_date(s: str) -> Optional[datetime]:
    for fmt in [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ]:
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return dt.replace(tzinfo=None)
        except Exception:
            continue
    return None


def _is_market_relevant(text: str) -> bool:
    keywords = [
        "NEPSE", "STOCK MARKET", "SHARE MARKET", "BULL", "BEAR",
        "DIVIDEND", "BONUS", "RIGHT SHARE", "IPO", "FPO",
        "NABIL", "NTC", "UPPER", "ADBL", "GBIME",
        "INTEREST RATE", "NRB", "RASTRA BANK", "INFLATION",
        "GDP", "REMITTANCE", "BANKING SECTOR",
    ]
    return any(k in text for k in keywords)


POSITIVE_WORDS = {
    "profit", "growth", "increase", "surge", "record", "high", "bull",
    "rally", "gain", "rise", "bonus", "dividend", "strong", "positive",
    "recovery", "expansion", "outperform", "beat", "upgrade", "buy",
    "accumulate", "upside", "opportunity", "optimistic", "robust",
}
NEGATIVE_WORDS = {
    "loss", "decline", "fall", "drop", "bear", "crash", "weak", "negative",
    "default", "risk", "concern", "worry", "sell", "downgrade", "cut",
    "problem", "issue", "crisis", "slow", "contraction", "miss", "disappoint",
    "fear", "uncertainty", "volatile", "correction", "plunge", "slump",
}


def _keyword_sentiment(text: str) -> Tuple[float, str]:
    words = re.findall(r"\b[a-z]+\b", text.lower())
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0, "neutral"
    score = (pos - neg) / total
    label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
    return round(score, 3), label


def get_aggregate_sentiment(symbol: str, days_back: int = 7) -> Dict[str, Any]:
    """
    Returns aggregate sentiment signal:
    {score: float [-1..1], label: str, count: int, articles: list}
    """
    articles = fetch_news_sentiment(symbol, days_back)
    if not articles:
        return {"score": 0.0, "label": "neutral", "count": 0, "articles": []}

    scores = [a["raw_score"] for a in articles]
    avg_score = float(np.mean(scores)) if scores else 0.0
    label = "positive" if avg_score > 0.1 else "negative" if avg_score < -0.1 else "neutral"

    return {
        "score": round(avg_score, 3),
        "label": label,
        "count": len(articles),
        "articles": articles[:10],  # top 10
    }