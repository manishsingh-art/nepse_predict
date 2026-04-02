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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import io
import requests
import pandas as pd
import numpy as np
import difflib

from cache import CACHE_DIR, get_cache_path, is_cache_fresh, load_json, load_pickle, save_json

logger = logging.getLogger(__name__)

# Some Windows/Python environments lack CA bundle for nepalstock.com; allow verify=False
# requests in controlled places and silence urllib3 warnings to keep logs clean.
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# ── Cache ──────────────────────────────────────────────────────────────────────
COMPANY_CACHE = get_cache_path("companies_cache.csv")
LEGACY_COMPANY_CACHE = Path(os.path.expanduser("~")) / ".nepse_cache" / "companies_cache.csv"
SYMBOLS_CACHE_FILE = get_cache_path("symbols_cache.json")
COMPANY_IDS_CACHE_FILE = get_cache_path("company_ids_cache.json")
SYMBOLS_CACHE_TTL_S = 24 * 60 * 60
COMPANY_CACHE_TTL_S = 24 * 60 * 60
PRICE_CACHE_TTL_S = 60 * 60
LIVE_PRICE_CACHE_TTL_S = 120
FLOORSHEET_CACHE_TTL_S = 15 * 60
NEWS_CACHE_TTL_S = 2 * 60 * 60
MARKET_LIVE_CACHE_TTL_S = 180

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

NEPALSTOCK_SECURITY_LIST = "https://www.nepalstock.com/api/nots/security"
NEPALSTOCK_SECURITY_LIST_FALLBACK = "https://nepalstock.com.np/api/nots/security?nonDelisted=true"
MEROLAGANI_COMPANY_LIST = "https://merolagani.com/CompanyList.aspx"
MEROLAGANI_AUTOSUGGEST = "https://merolagani.com/handlers/AutoSuggestHandler.ashx?type=Company"
# Listed companies + bonds/debentures: one row per security with symbol + legal name
_MEROLAGANI_LIST_ROW = re.compile(
    r"href=['\"]/CompanyDetail\.aspx\?symbol=([^'\"]+)['\"][^>]*>\s*([^<]+?)\s*</a>\s*</td>\s*"
    r"<td[^>]*>\s*([^<]+?)\s*</td>",
    re.IGNORECASE | re.DOTALL,
)
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
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _history_cache_path(symbol: str, years: int) -> Path:
    return get_cache_path(f"{symbol.upper()}_{years}y_history.pkl")


def _legacy_history_cache_path(symbol: str, years: int) -> Path:
    return Path(os.path.expanduser("~")) / ".nepse_cache" / "price_cache" / f"{symbol.upper()}_{years}y.pkl"


def _live_price_cache_path(symbol: str) -> Path:
    return get_cache_path(f"{symbol.upper()}_live_price.json")


def _floorsheet_cache_path(symbol: str) -> Path:
    return get_cache_path(f"{symbol.upper()}_floorsheet.pkl")


def _news_cache_path(symbol: str, days_back: int) -> Path:
    return get_cache_path(f"{symbol.upper()}_{days_back}d_sentiment.json")


def _market_live_cache_path() -> Path:
    return get_cache_path("market_live_status.json")


def _load_pickle_with_fallback(primary: Path, legacy: Optional[Path] = None, max_age_seconds: Optional[int] = None):
    payload = load_pickle(primary, max_age_seconds=max_age_seconds)
    if payload is not None:
        return payload, "cache:fresh"
    if primary.exists():
        payload = load_pickle(primary, max_age_seconds=None)
        if payload is not None:
            return payload, "cache:stale"
    if legacy is not None and legacy.exists():
        payload = load_pickle(legacy, max_age_seconds=max_age_seconds)
        if payload is not None:
            return payload, "cache:fresh-legacy"
        payload = load_pickle(legacy, max_age_seconds=None)
        if payload is not None:
            return payload, "cache:stale-legacy"
    return None, None


def _coerce_return(payload, source: str, return_source: bool):
    return (payload, source) if return_source else payload


def _fetch_rss_articles(feed_url: str, symbol: str, cutoff: datetime) -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []
    try:
        r = robust_request(feed_url, timeout=10, retries=2)
        if r is None:
            return articles
        _parse_rss_feed(r.text, symbol, cutoff, articles)
    except Exception:
        return []
    return articles


def _fuzzy_match(query: str, symbols: List[str]) -> Tuple[Optional[str], float]:
    """
    Return best (match, score 0..100) using lightweight difflib ratio.
    """
    q = (query or "").strip().upper()
    if not q or not symbols:
        return None, 0.0
    best_sym, best_score = None, 0.0
    for s in symbols:
        score = difflib.SequenceMatcher(a=q, b=str(s).upper()).ratio() * 100.0
        if score > best_score:
            best_score = score
            best_sym = str(s).upper()
    return best_sym, float(best_score)


def _read_symbols_cache() -> Optional[Dict[str, str]]:
    try:
        if not SYMBOLS_CACHE_FILE.exists():
            return None
        age = time.time() - SYMBOLS_CACHE_FILE.stat().st_mtime
        if age > SYMBOLS_CACHE_TTL_S:
            logger.info("Symbols cache miss (expired)")
            return None
        payload = json.loads(SYMBOLS_CACHE_FILE.read_text(encoding="utf-8"))
        items = payload.get("symbols", payload)
        if not isinstance(items, dict):
            return None
        out = {str(k).upper(): str(v) for k, v in items.items() if str(k).strip()}
        logger.info("Symbols cache hit")
        return out if out else None
    except Exception:
        return None


def _write_symbols_cache(symbols: Dict[str, str]) -> None:
    try:
        SYMBOLS_CACHE_FILE.write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now().isoformat(timespec="seconds"),
                    "source": "merged",
                    "count": len(symbols),
                    "symbols": symbols,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def _sanitize_display_name(name: str, sym: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "").strip())
    if not n or n.lower() in ("nan", "none", "nat"):
        return sym
    return n


def _read_company_ids_cache() -> Dict[str, str]:
    """Return cached {SYMBOL: merolagani_id} mapping, or empty dict if missing/stale."""
    try:
        if not COMPANY_IDS_CACHE_FILE.exists():
            return {}
        age = time.time() - COMPANY_IDS_CACHE_FILE.stat().st_mtime
        if age > SYMBOLS_CACHE_TTL_S:
            return {}
        payload = json.loads(COMPANY_IDS_CACHE_FILE.read_text(encoding="utf-8"))
        ids = payload.get("ids", payload)
        if not isinstance(ids, dict):
            return {}
        return {str(k).upper(): str(v) for k, v in ids.items() if str(k).strip()}
    except Exception:
        return {}


def _write_company_ids_cache(ids: Dict[str, str]) -> None:
    try:
        COMPANY_IDS_CACHE_FILE.write_text(
            json.dumps(
                {
                    "fetched_at": datetime.now().isoformat(timespec="seconds"),
                    "count": len(ids),
                    "ids": ids,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_company_id(symbol: str) -> str:
    """
    Return the MeroLagani internal company ID for `symbol`, or empty string if unknown.
    Used to construct NepalStock history API URLs.
    """
    ids = _read_company_ids_cache()
    return ids.get(symbol.strip().upper(), "")


def _fetch_symbols_autosuggest() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Fetch ALL listed NEPSE securities from MeroLagani's AutoSuggest JSON endpoint.
    Returns (symbols_dict, ids_dict) where:
      symbols_dict = {SYMBOL: company_name}
      ids_dict     = {SYMBOL: merolagani_internal_id}

    This endpoint returns 5 000+ securities (equities, bonds, debentures, mutual funds)
    in a single lightweight JSON call, far more complete than the HTML CompanyList scrape.
    Response item format: {"l": "SYMBOL (Company Name)", "v": "1234", "d": "SYMBOL"}
    """
    symbols: Dict[str, str] = {}
    ids: Dict[str, str] = {}
    r = robust_request(
        MEROLAGANI_AUTOSUGGEST,
        timeout=30,
        retries=2,
        headers={**HEADERS, "Referer": "https://merolagani.com/"},
    )
    if r is None or r.status_code != 200:
        return symbols, ids
    try:
        items = r.json()
        if not isinstance(items, list):
            return symbols, ids
        for item in items:
            sym = str(item.get("d") or "").strip().upper()
            if not sym:
                continue
            label = str(item.get("l") or "").strip()
            # label format: "SYMBOL (Company Name)" — extract company name
            m = re.match(r"[^\(]+\((.+)\)\s*$", label)
            name = _sanitize_display_name(m.group(1) if m else label, sym)
            company_id = str(item.get("v") or "").strip()
            symbols[sym] = name
            if company_id:
                ids[sym] = company_id
    except Exception as exc:
        logger.debug("AutoSuggest fetch error: %s", exc)
    return symbols, ids


def _fetch_symbols_merolagani_company_list() -> Dict[str, str]:
    """
    Full NEPSE-listed securities (equities, bonds, debentures) from MeroLagani.
    Official NEPSE JSON often returns 401 without browser/session auth; this source is public HTML.
    """
    out: Dict[str, str] = {}
    r = robust_request(
        MEROLAGANI_COMPANY_LIST,
        timeout=45,
        retries=2,
        headers={**HEADERS, "Referer": "https://merolagani.com/"},
    )
    if r is None or r.status_code != 200:
        return out
    for m in _MEROLAGANI_LIST_ROW.finditer(r.text):
        raw_sym = m.group(1).strip()
        sym = raw_sym.upper()
        if not sym:
            continue
        name = _sanitize_display_name(m.group(3), sym)
        out[sym] = name
    return out


def _fetch_symbols_nepse_api() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for url in (NEPALSTOCK_SECURITY_LIST, NEPALSTOCK_SECURITY_LIST_FALLBACK):
        r = robust_request(url, timeout=20, retries=2, verify=False)
        if r is None or r.status_code != 200:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        items = None
        if isinstance(data, dict):
            if isinstance(data.get("body"), list):
                items = data["body"]
            elif isinstance(data.get("content"), list):
                items = data["content"]
            elif isinstance(data.get("data"), list):
                items = data["data"]
        if items is None:
            items = data if isinstance(data, list) else None
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            sym = (item.get("symbol") or item.get("stockSymbol") or "").strip().upper()
            if not sym:
                continue
            name = (
                item.get("securityName")
                or item.get("companyName")
                or item.get("securityNameEnglish")
                or item.get("stock_name")
                or sym
            )
            out[sym] = _sanitize_display_name(name, sym)
        if out:
            break
    return out


def fetch_nepse_symbols(force_refresh: bool = False) -> Dict[str, str]:
    """
    Fetch ALL listed NEPSE securities (equities, bonds, debentures, mutual funds).

    Primary source: MeroLagani AutoSuggest JSON API — returns 5 000+ securities in one
    lightweight call, including newly listed stocks. Also extracts internal company IDs
    used for NepalStock history API lookups.
    Fallback: MeroLagani CompanyList HTML, then NEPSE API, then StockQuote/ShareSansar.
    Cache: symbols_cache.json + company_ids_cache.json, TTL 24 h.
    Incomplete caches (< 200 symbols) are ignored so a bad snapshot auto-refreshes.
    """
    if not force_refresh:
        cached = _read_symbols_cache()
        if cached and len(cached) >= 200:
            logger.info("Total symbols loaded: %d", len(cached))
            return cached
        if cached is not None and len(cached) < 200:
            logger.info("Symbols cache miss (incomplete snapshot: %d symbols)", len(cached))
        elif cached is None:
            logger.info("Symbols cache miss (no cache)")

    symbols: Dict[str, str] = {}
    ids: Dict[str, str] = {}

    # ── Primary: AutoSuggest JSON (fastest, most complete — 5 000+ entries) ───
    as_syms, as_ids = _fetch_symbols_autosuggest()
    if as_syms:
        symbols.update(as_syms)
        ids.update(as_ids)
        logger.info(
            "Symbols source: MeroLagani AutoSuggest (%d securities, %d with IDs)",
            len(as_syms),
            len(as_ids),
        )

    # ── Fallback: CompanyList HTML (if AutoSuggest is unreachable) ─────────────
    if len(symbols) < 200:
        ml = _fetch_symbols_merolagani_company_list()
        if ml:
            added = 0
            for sym, name in ml.items():
                if sym not in symbols:
                    symbols[sym] = name
                    added += 1
            logger.info("Symbols source: MeroLagani CompanyList (+%d new securities)", added)

    # ── Merge: official NEPSE API (may fill name gaps; often returns 401) ─────
    api_syms = _fetch_symbols_nepse_api()
    if api_syms:
        added = 0
        for sym, name in api_syms.items():
            if sym not in symbols:
                symbols[sym] = name
                added += 1
            elif symbols[sym] == sym and name != sym:
                symbols[sym] = name
        if added:
            logger.info("Symbols source: NEPSE API merged (+%d new)", added)

    # ── Last resort: local CSV / StockQuote regex / ShareSansar ───────────────
    if len(symbols) < 100 and COMPANY_CACHE.exists():
        try:
            df = pd.read_csv(COMPANY_CACHE)
            if "symbol" in df.columns:
                for _, row in df.iterrows():
                    sym = str(row.get("symbol", "")).strip().upper()
                    if not sym:
                        continue
                    raw = row.get("name", sym)
                    name = sym if pd.isna(raw) else _sanitize_display_name(raw, sym)
                    symbols.setdefault(sym, name)
            logger.info("Symbols source: disk companies_cache.csv fallback (%d total)", len(symbols))
        except Exception:
            pass

    if len(symbols) < 100:
        try:
            r = robust_request("https://merolagani.com/StockQuote.aspx", timeout=15, retries=2)
            if r and r.status_code == 200:
                for sym in re.findall(r"symbol=([A-Z0-9][A-Z0-9./+-]{1,14})[\"']", r.text):
                    symbols.setdefault(str(sym).upper(), str(sym).upper())
        except Exception:
            pass

    if len(symbols) < 100:
        try:
            r = robust_request("https://www.sharesansar.com/today-share-price", timeout=15, retries=2)
            if r and r.status_code == 200:
                tables = pd.read_html(io.StringIO(r.text))
                for t in tables:
                    cols = [str(c).lower() for c in t.columns]
                    if any("symbol" in c or "ticker" in c for c in cols):
                        t.columns = cols
                        sym_col = next(c for c in cols if "symbol" in c or "ticker" in c)
                        for sym in t[sym_col].astype(str).str.upper():
                            if 2 <= len(sym) <= 16 and sym != "NAN":
                                symbols.setdefault(sym, sym)
        except Exception:
            pass

    if symbols:
        _write_symbols_cache(symbols)
    if ids:
        _write_company_ids_cache(ids)
    logger.info("Total symbols loaded: %d", len(symbols))
    return symbols


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
    try:
        if COMPANY_CACHE.exists() and is_cache_fresh(COMPANY_CACHE, COMPANY_CACHE_TTL_S):
            df = pd.read_csv(COMPANY_CACHE)
            if "symbol" in df.columns and len(df) > 20:
                df["symbol"] = df["symbol"].astype(str).str.upper()
                return df
    except Exception:
        pass

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
                tables = pd.read_html(io.StringIO(r.text))
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
    for cache_file in (COMPANY_CACHE, LEGACY_COMPANY_CACHE):
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file)
                df["symbol"] = df["symbol"].astype(str).str.upper()
                if len(df) > 20:
                    return df
            except Exception:
                pass

    return _builtin_company_list()


def resolve_symbol(symbol: str) -> str:
    sym = (symbol or "").strip().upper()
    symbols = fetch_nepse_symbols()
    if sym in symbols:
        return sym

    match, score = _fuzzy_match(sym, list(symbols.keys()))
    if match and score > 90:
        return match

    # New listing handling: try direct price fetch, then add to cache
    try:
        price = fetch_live_price(sym)
        if price is not None and float(price) > 0:
            logger.info("NEW LISTING DETECTED")
            symbols[sym] = symbols.get(sym, sym)
            _write_symbols_cache(symbols)
            return sym
    except Exception:
        pass

    # Last-resort validation: try fetching any history (still "direct fetch")
    try:
        df = fetch_history(sym, years=1)
        if df is not None and not df.empty and len(df) >= 5:
            logger.info("NEW LISTING DETECTED")
            symbols[sym] = symbols.get(sym, sym)
            _write_symbols_cache(symbols)
            return sym
    except Exception:
        pass

    raise ValueError(f"Symbol {sym} not found (new listing?)")


def is_known_symbol_local(symbol: str) -> bool:
    """
    Fast local-only symbol check using only on-disk caches — never hits the network.
    Checks AutoSuggest symbols cache (1 600+ entries) and company_ids cache.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return False

    cached = _read_symbols_cache()
    if cached and sym in cached:
        return True

    ids = _read_company_ids_cache()
    if ids and sym in ids:
        return True

    for cache_file in (COMPANY_CACHE, LEGACY_COMPANY_CACHE):
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, usecols=["symbol"])
                if sym in set(df["symbol"].astype(str).str.upper()):
                    return True
            except Exception:
                pass

    return False


def _builtin_company_list() -> pd.DataFrame:
    """
    Returns an empty DataFrame — all symbol data now comes dynamically from
    the MeroLagani AutoSuggest API via fetch_nepse_symbols() / _fetch_symbols_autosuggest().
    This stub exists only for backward compatibility with callers.
    """
    return pd.DataFrame(columns=["symbol", "name", "sector", "id"])


# ── Historical Price Data ─────────────────────────────────────────────────────

def fetch_history(
    symbol: str,
    company_id: str = "",
    years: int = 5,
    return_source: bool = False,
    max_age_seconds: int = PRICE_CACHE_TTL_S,
    allow_company_lookup: bool = True,
) -> pd.DataFrame | Tuple[pd.DataFrame, str]:
    """
    Return cleaned OHLCV DataFrame for `symbol` covering `years` years.
    Tries sources in order: MeroLagani → NepalStock → ShareSansar → cache.
    """
    _ensure_dirs()
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365 * years)

    cache_file = _history_cache_path(symbol, years)
    legacy_cache_file = _legacy_history_cache_path(symbol, years)
    cached_df, cache_source = _load_pickle_with_fallback(
        cache_file,
        legacy=legacy_cache_file,
        max_age_seconds=max_age_seconds,
    )
    stale_df = None
    if cached_df is not None:
        logger.debug("Price cache hit for %s", symbol)
        if cache_source and cache_source.startswith("cache:fresh"):
            return _coerce_return(cached_df, cache_source, return_source)
        stale_df = cached_df

    # New listings may have very few rows; accept >= 5 from any source and
    # keep the best (most rows) across all sources before deciding.
    MIN_ROWS_SUFFICIENT = 20   # prefer sources with this many rows or more
    MIN_ROWS_ACCEPT = 5        # accept new-listing data with at least this many rows

    best_df: Optional[pd.DataFrame] = None
    best_source: str = ""

    def _candidate(candidate_df: Optional[pd.DataFrame], source: str) -> bool:
        """Return True if this candidate is good enough to use immediately."""
        nonlocal best_df, best_source
        if candidate_df is None or len(candidate_df) < MIN_ROWS_ACCEPT:
            return False
        if best_df is None or len(candidate_df) > len(best_df):
            best_df = candidate_df
            best_source = source
        return len(candidate_df) >= MIN_ROWS_SUFFICIENT

    df = _fetch_merolagani(symbol, start_dt, end_dt)
    if _candidate(df, "merolagani") and best_df is not None:
        best_df.to_pickle(cache_file)
        return _coerce_return(best_df, best_source, return_source)

    if allow_company_lookup and not company_id:
        # Fast path: use the ID cached from the AutoSuggest API (no HTTP round-trip)
        company_id = get_company_id(symbol)
        if company_id:
            logger.debug("Company ID for %s resolved from AutoSuggest cache: %s", symbol, company_id)
    if allow_company_lookup and not company_id:
        try:
            companies = fetch_company_list()
            m = companies[companies["symbol"].str.upper() == symbol.upper()]
            if not m.empty:
                company_id = str(m.iloc[0].get("id", ""))
        except Exception:
            pass

    if company_id:
        df = _fetch_nepalstock(company_id, start_dt, end_dt)
        if _candidate(df, "nepalstock") and best_df is not None and len(best_df) >= MIN_ROWS_SUFFICIENT:
            best_df.to_pickle(cache_file)
            return _coerce_return(best_df, best_source, return_source)

    df = _fetch_sharesansar(symbol)
    _candidate(df, "sharesansar")

    # Return the best result we found (even if it's a new listing with few rows)
    if best_df is not None:
        best_df.to_pickle(cache_file)
        if len(best_df) < MIN_ROWS_SUFFICIENT:
            logger.warning(
                "NEW LISTING: %s has only %d trading rows — ML features will be limited.",
                symbol, len(best_df),
            )
        return _coerce_return(best_df, best_source, return_source)

    # Try loading stale cache as last resort
    if stale_df is not None and len(stale_df) >= MIN_ROWS_ACCEPT:
        logger.warning("Using stale cache for %s", symbol)
        return _coerce_return(stale_df, cache_source or "cache:stale", return_source)

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


def fetch_live_price(
    symbol: str,
    return_source: bool = False,
    max_age_seconds: int = LIVE_PRICE_CACHE_TTL_S,
) -> Optional[float] | Tuple[Optional[float], str]:
    """Scrape the latest LTP for a symbol using multiple sources (real-time)."""
    sym = symbol.upper().strip()
    cache_file = _live_price_cache_path(sym)
    cached_payload = load_json(cache_file, max_age_seconds=max_age_seconds)
    stale_payload = load_json(cache_file, max_age_seconds=None) if cache_file.exists() else None

    def _cached_result(payload: Optional[dict], label: str):
        if isinstance(payload, dict):
            price = payload.get("price")
            if price is not None:
                return _coerce_return(float(price), label, return_source)
        return None

    # Source 1: NEPSE Official Live JSON
    try:
        url = "https://nepalstock.com.np/api/nots/market/active-securities"
        r = robust_request(url, timeout=5, verify=False)
        if r:
            data = r.json()
            for item in data:
                if item.get("symbol") == sym:
                    price = item.get("lastTradedPrice") or item.get("closePrice")
                    if price and float(price) > 0:
                        save_json(cache_file, {"price": float(price), "source": "nepalstock_live"})
                        return _coerce_return(float(price), "nepalstock_live", return_source)
    except Exception:
        pass

    # Source 2: MeroLagani Today's Share Price
    try:
        r = robust_request("https://merolagani.com/LatestMarket.aspx", timeout=8, verify=False)
        if r:
            tables = pd.read_html(io.StringIO(r.text))
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
                        if price > 0:
                            save_json(cache_file, {"price": float(price), "source": "merolagani_live"})
                            return _coerce_return(float(price), "merolagani_live", return_source)
    except Exception:
        pass

    # Source 3: ShareSansar Today Share Price
    try:
        r = robust_request("https://www.sharesansar.com/today-share-price", timeout=8, verify=False)
        if r:
            tables = pd.read_html(io.StringIO(r.text))
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
                        if price > 0:
                            save_json(cache_file, {"price": float(price), "source": "sharesansar_live"})
                            return _coerce_return(float(price), "sharesansar_live", return_source)
    except Exception:
        pass

    fresh = _cached_result(cached_payload, "cache:fresh-live")
    if fresh is not None:
        return fresh
    stale = _cached_result(stale_payload, "cache:stale-live")
    if stale is not None:
        return stale
    return _coerce_return(None, "unavailable", return_source)


def fetch_market_live_status(
    return_source: bool = False,
    max_age_seconds: int = MARKET_LIVE_CACHE_TTL_S,
) -> bool | Tuple[bool, str]:
    """
    Detect whether any live market feed exists today.
    This is intentionally lightweight and avoids static holiday rules.
    """
    cache_file = _market_live_cache_path()
    cached = load_json(cache_file, max_age_seconds=max_age_seconds)
    if isinstance(cached, dict):
        return _coerce_return(
            bool(cached.get("has_live_data", False)),
            str(cached.get("source") or "cache:fresh-market"),
            return_source,
        )

    try:
        r = robust_request(
            "https://nepalstock.com.np/api/nots/market/active-securities",
            timeout=2,
            retries=1,
            verify=False,
        )
        if r:
            data = r.json()
            has_live_data = bool(
                isinstance(data, list)
                and any((item.get("lastTradedPrice") or item.get("closePrice")) for item in data if isinstance(item, dict))
            )
            save_json(cache_file, {"has_live_data": has_live_data, "source": "nepalstock_active_securities"})
            return _coerce_return(has_live_data, "nepalstock_active_securities", return_source)
    except Exception:
        pass

    cached = load_json(cache_file, max_age_seconds=None)
    if isinstance(cached, dict):
        return _coerce_return(
            bool(cached.get("has_live_data", False)),
            str(cached.get("source") or "cache:stale-market"),
            return_source,
        )

    return _coerce_return(False, "live_feed_unavailable", return_source)


def _fetch_sharesansar(symbol: str) -> Optional[pd.DataFrame]:
    try:
        r = robust_request(f"https://www.sharesansar.com/company/{symbol.lower()}")
        if r is None: return None
        tables = pd.read_html(io.StringIO(r.text))
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

def fetch_floorsheet(
    symbol: str,
    return_source: bool = False,
    max_age_seconds: int = FLOORSHEET_CACHE_TTL_S,
) -> Optional[pd.DataFrame] | Tuple[Optional[pd.DataFrame], str]:
    """
    Fetch the latest floorsheet (intraday transactions) for a symbol 
    from ShareSansar. Used to detect buyer/seller concentration.
    """
    symbol = symbol.upper()
    cache_file = _floorsheet_cache_path(symbol)
    cached_df, cache_source = _load_pickle_with_fallback(cache_file, max_age_seconds=max_age_seconds)
    if cached_df is not None and cache_source and cache_source.startswith("cache:fresh"):
        return _coerce_return(cached_df, cache_source, return_source)
    try:
        url = f"https://www.sharesansar.com/floorsheet?symbol={symbol}"
        r = robust_request(url, headers={**HEADERS, "Referer": "https://www.sharesansar.com/"})
        if r is None:
            if cached_df is not None:
                return _coerce_return(cached_df, cache_source or "cache:stale", return_source)
            return _coerce_return(None, "unavailable", return_source)
        
        tables = pd.read_html(io.StringIO(r.text))
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
                t.to_pickle(cache_file)
                return _coerce_return(t, "sharesansar_floorsheet", return_source)
    except Exception as e:
        logger.debug("Floorsheet fetch error for %s: %s", symbol, e)
    if cached_df is not None:
        return _coerce_return(cached_df, cache_source or "cache:stale", return_source)
    return _coerce_return(None, "unavailable", return_source)


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


def fetch_news_sentiment(
    symbol: str,
    days_back: int = 7,
    return_source: bool = False,
    max_age_seconds: int = NEWS_CACHE_TTL_S,
) -> List[Dict[str, Any]] | Tuple[List[Dict[str, Any]], str]:
    """
    Fetch news articles mentioning `symbol` from RSS feeds.
    Returns list of dicts: {title, published, source, raw_score, label}
    where raw_score is basic keyword polarity (-1..+1).
    """
    _ensure_dirs()
    cache_file = _news_cache_path(symbol, days_back)

    cached_articles = load_json(cache_file, max_age_seconds=max_age_seconds)
    if isinstance(cached_articles, list):
        return _coerce_return(cached_articles, "cache:fresh", return_source)

    stale_articles = load_json(cache_file, max_age_seconds=None) if cache_file.exists() else None

    cutoff = datetime.now() - timedelta(days=days_back)
    articles = []

    with ThreadPoolExecutor(max_workers=min(4, len(RSS_FEEDS))) as executor:
        futures = [executor.submit(_fetch_rss_articles, feed_url, symbol, cutoff) for feed_url in RSS_FEEDS]
        for future in futures:
            try:
                articles.extend(future.result())
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
        save_json(cache_file, unique_articles)
    except Exception:
        pass

    if unique_articles:
        return _coerce_return(unique_articles, "rss_live", return_source)
    if isinstance(stale_articles, list):
        return _coerce_return(stale_articles, "cache:stale", return_source)
    return _coerce_return([], "unavailable", return_source)


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