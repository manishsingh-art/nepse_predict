#!/usr/bin/env python3
"""
NEPSE prediction system (daily runner)
=====================================
End-to-end pipeline:
- Fetch historical + latest OHLCV data for a symbol (via fetcher.py).
- Maintain a growing CSV dataset (append/dedupe by date).
- Feature engineering (MAs, returns, volume trends, volatility).
- Train Linear Regression (baseline) + Random Forest (non-linear).
- Forecast next 5–10 business days (recursive multi-step).
- Ask local Ollama (mistral) for narrative + risk + signals.
- Save a daily report (json + txt) and a prediction plot (png).

Note: This is educational and not financial advice.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fetcher import fetch_company_list, fetch_history


DATE_COL = "date"


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    n_train: int
    n_valid: int


@dataclass
class ForecastRow:
    date: str
    predicted_close_lr: float
    predicted_close_rf: float
    predicted_close_blend: float
    expected_change_pct_from_last: float


@dataclass
class PredictionComparison:
    target_date: str
    actual_close: float
    predicted_close: float
    diff: float
    error_pct: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # tolerate common header variants
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl == "date":
            rename[c] = "date"
        elif cl == "open":
            rename[c] = "open"
        elif cl == "high":
            rename[c] = "high"
        elif cl == "low":
            rename[c] = "low"
        elif cl in {"close", "ltp", "last"}:
            rename[c] = "close"
        elif cl in {"volume", "qty", "quantity"}:
            rename[c] = "volume"
    if rename:
        df = df.rename(columns=rename)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[DATE_COL, "close"]).sort_values(DATE_COL)
    df = df.drop_duplicates(subset=[DATE_COL]).reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df


def load_existing_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # tolerate common cases where date column differs
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols and cols["date"] != "date":
        df = df.rename(columns={cols["date"]: "date"})
    return _standardize_ohlcv(df)


def merge_append_save(existing: Optional[pd.DataFrame], new_df: pd.DataFrame, out_csv: str) -> pd.DataFrame:
    new_df = _standardize_ohlcv(new_df)
    if existing is None or existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = _standardize_ohlcv(merged)
    ensure_dir(os.path.dirname(out_csv))
    merged.to_csv(out_csv, index=False)
    return merged


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    close = df["close"].astype(float)
    vol = df["volume"].astype(float)

    # Returns
    df["ret_1d"] = close.pct_change(fill_method=None)
    df["ret_5d"] = close.pct_change(5, fill_method=None)
    df["ret_10d"] = close.pct_change(10, fill_method=None)

    # Moving averages / distance to MA
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = close.rolling(w).mean()
        df[f"close_to_sma_{w}"] = close / (df[f"sma_{w}"] + 1e-9) - 1.0

    # Volatility (rolling std of returns)
    df["volatility_10d"] = df["ret_1d"].rolling(10).std()
    df["volatility_20d"] = df["ret_1d"].rolling(20).std()

    # Volume trends
    df["vol_sma_5"] = vol.rolling(5).mean()
    df["vol_sma_20"] = vol.rolling(20).mean()
    df["vol_ratio_5_20"] = df["vol_sma_5"] / (df["vol_sma_20"] + 1e-9)
    df["vol_change_1d"] = vol.pct_change(fill_method=None)

    # Lag features (price level + returns)
    for lag in [1, 2, 3, 5, 10]:
        df[f"close_lag_{lag}"] = close.shift(lag)
        df[f"ret_lag_{lag}"] = df["ret_1d"].shift(lag)

    # Calendar-ish
    df["dow"] = df[DATE_COL].dt.dayofweek  # 0=Mon
    return df


def make_supervised(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Predict next day's close.
    Returns X, y, feature_df_aligned (same index as X/y).
    """
    feat = add_features(df)
    feat["target_next_close"] = feat["close"].shift(-1)
    feat = feat.dropna(subset=["target_next_close"]).reset_index(drop=True)

    y = feat["target_next_close"].astype(float)
    drop_cols = {DATE_COL, "target_next_close"}
    # Keep ohlcv and engineered features; date is removed from model features.
    X = feat[[c for c in feat.columns if c not in drop_cols]]
    return X, y, feat


def time_split(X: pd.DataFrame, y: pd.Series, valid_frac: float = 0.2) -> Tuple:
    n = len(X)
    n_valid = max(20, int(n * valid_frac))
    n_valid = min(n_valid, n - 1) if n > 1 else 0
    n_train = n - n_valid
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_valid, y_valid = X.iloc[n_train:], y.iloc[n_train:]
    return X_train, X_valid, y_train, y_valid


def build_models(feature_names: List[str]) -> Dict[str, Pipeline]:
    numeric_features = feature_names
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", numeric, numeric_features)],
        remainder="drop",
    )

    # Linear regression baseline (SGD is robust + fast on most machines)
    lr = Pipeline(
        steps=[
            ("pre", pre),
            ("model", SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4, max_iter=5000, tol=1e-3, random_state=42)),
        ]
    )
    rf = Pipeline(
        steps=[
            ("pre", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=30,
                    random_state=42,
                    n_jobs=1,
                    min_samples_leaf=2,
                    max_depth=12,
                ),
            ),
        ]
    )
    return {"lr": lr, "rf": rf}


def fit_and_score(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> ModelMetrics:
    t0 = datetime.now()
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = float(mean_absolute_error(y_valid, preds))
    rmse = float(np.sqrt(mean_squared_error(y_valid, preds)))
    _ = t0  # keep scope explicit for debugging if needed
    return ModelMetrics(mae=mae, rmse=rmse, n_train=len(X_train), n_valid=len(X_valid))


def next_business_days(last_date: pd.Timestamp, n: int) -> List[pd.Timestamp]:
    # Nepal holidays are not encoded; this approximates with Mon–Fri.
    start = last_date + pd.offsets.BDay(1)
    return list(pd.bdate_range(start=start, periods=n))


def _latest_volume_proxy(df: pd.DataFrame) -> float:
    if "volume" not in df.columns:
        return float("nan")
    s = pd.to_numeric(df["volume"], errors="coerce")
    v = float(s.tail(20).median()) if s.notna().any() else float("nan")
    return v


def recursive_forecast(
    history_df: pd.DataFrame,
    lr_model: Pipeline,
    rf_model: Pipeline,
    horizon: int,
) -> List[ForecastRow]:
    df = _standardize_ohlcv(history_df)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    last_close = float(df["close"].iloc[-1])
    vol_proxy = _latest_volume_proxy(df)

    future_dates = next_business_days(df[DATE_COL].iloc[-1], horizon)
    synthetic = df.copy()
    out: List[ForecastRow] = []

    for dt in future_dates:
        # Create a new row placeholder for feature computation.
        new_row = {
            DATE_COL: dt,
            "open": np.nan,
            "high": np.nan,
            "low": np.nan,
            "close": np.nan,  # to be predicted
            "volume": vol_proxy,
        }
        synthetic = pd.concat([synthetic, pd.DataFrame([new_row])], ignore_index=True)

        # Build features on synthetic history; take last row features to predict next close.
        X_all, y_all, feat = make_supervised(synthetic.dropna(subset=["close"]).copy() if synthetic["close"].notna().any() else synthetic.copy())

        # We need features for the last *known* row to predict the next close.
        # Since target is shift(-1), predicting next close uses last available feature row.
        # Here, we use features computed from synthetic with close filled for prior steps.
        feat2 = add_features(synthetic.copy())
        feat_last = feat2.iloc[-1:].copy()
        # fill close with last known/pred close for feature coherence
        if pd.isna(feat_last["close"].iloc[0]):
            feat_last.loc[feat_last.index[0], "close"] = synthetic["close"].dropna().iloc[-1]

        feat_last = add_features(pd.concat([synthetic.iloc[:-1].copy(), feat_last], ignore_index=True)).iloc[-1:]
        # Align to model feature columns (same as training X columns)
        feature_cols = [c for c in X_all.columns]
        X_pred = feat_last[feature_cols]

        pred_lr = float(lr_model.predict(X_pred)[0])
        pred_rf = float(rf_model.predict(X_pred)[0])
        # Blend (RF typically better on non-linearities; keep LR as anchor)
        pred_blend = 0.35 * pred_lr + 0.65 * pred_rf

        # Respect NEPSE circuit filter (±10% from previous close) as a risk guardrail.
        cap_hi = last_close * 1.10
        cap_lo = last_close * 0.90
        pred_blend = float(np.clip(pred_blend, cap_lo, cap_hi))

        # Set synthetic close for next-step features.
        synthetic.loc[synthetic.index[-1], "close"] = pred_blend

        exp_change = (pred_blend - last_close) / (last_close + 1e-9) * 100.0
        out.append(
            ForecastRow(
                date=dt.strftime("%Y-%m-%d"),
                predicted_close_lr=round(pred_lr, 2),
                predicted_close_rf=round(pred_rf, 2),
                predicted_close_blend=round(pred_blend, 2),
                expected_change_pct_from_last=round(exp_change, 2),
            )
        )
        last_close = pred_blend
    return out


def classify_trend(df: pd.DataFrame, forecast: List[ForecastRow]) -> Tuple[str, str]:
    """
    Simple, realistic trend classification:
    - Look at last 20 sessions return + predicted 5-day return.
    """
    df = _standardize_ohlcv(df)
    df = df.sort_values(DATE_COL)
    last = float(df["close"].iloc[-1])
    prev20 = float(df["close"].iloc[-20]) if len(df) >= 20 else float(df["close"].iloc[0])
    ret20 = (last - prev20) / (prev20 + 1e-9) * 100.0

    if forecast:
        idx = min(4, len(forecast) - 1)
        f5 = float(forecast[idx].predicted_close_blend)
        ret_pred = (f5 - last) / (last + 1e-9) * 100.0
    else:
        ret_pred = 0.0

    score = 0
    score += 1 if ret20 > 0 else -1
    score += 1 if ret_pred > 0 else -1

    if score >= 2:
        return "bullish", f"20d={ret20:+.2f}%, pred5d={ret_pred:+.2f}%"
    if score <= -2:
        return "bearish", f"20d={ret20:+.2f}%, pred5d={ret_pred:+.2f}%"
    return "neutral", f"20d={ret20:+.2f}%, pred5d={ret_pred:+.2f}%"


def rule_based_recommendation(df: pd.DataFrame, forecast: List[ForecastRow]) -> Tuple[str, str]:
    """
    Conservative recommendation using forecast + volatility.
    """
    df = add_features(_standardize_ohlcv(df))
    last = df.iloc[-1]
    last_close = float(last["close"])
    vol20 = float(last.get("volatility_20d", np.nan))
    vol_label = "high" if (not np.isnan(vol20) and vol20 > 0.03) else "normal"

    if not forecast:
        return "hold", "Insufficient forecast horizon."

    f5 = float(forecast[min(4, len(forecast) - 1)].predicted_close_blend)
    exp = (f5 - last_close) / (last_close + 1e-9) * 100.0

    if exp > 2.0 and vol_label != "high":
        return "buy", f"Predicted +{exp:.2f}% over ~5 sessions with {vol_label} volatility."
    if exp < -2.0:
        return "sell", f"Predicted {exp:.2f}% over ~5 sessions; downside risk dominates."
    return "hold", f"Predicted {exp:+.2f}% over ~5 sessions; edge not strong vs noise (vol={vol_label})."


def call_ollama(prompt: str, model: str = "mistral", url: str = "http://localhost:11434/api/generate", timeout: int = 60) -> Optional[str]:
    try:
        import requests

        payload = {"model": model, "prompt": prompt, "stream": False}
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("response")
    except Exception:
        return None


def build_ollama_prompt(symbol: str, df: pd.DataFrame, forecast: List[ForecastRow], lr_metrics: ModelMetrics, rf_metrics: ModelMetrics) -> str:
    df = _standardize_ohlcv(df).sort_values(DATE_COL)
    tail = df.tail(25).copy()
    tail[DATE_COL] = tail[DATE_COL].dt.strftime("%Y-%m-%d")

    return (
        "You are a cautious financial analyst. Do NOT hallucinate news.\n"
        "Use only the provided data + predictions. Output concise bullets.\n\n"
        f"SYMBOL: {symbol}\n"
        f"LAST_CLOSE: {df['close'].iloc[-1]:.2f}\n"
        f"DATA_TAIL (last 25 rows, CSV):\n{tail.to_csv(index=False)}\n"
        f"MODEL_METRICS:\n"
        f"- LinearRegression: MAE={lr_metrics.mae:.3f}, RMSE={lr_metrics.rmse:.3f}\n"
        f"- RandomForest    : MAE={rf_metrics.mae:.3f}, RMSE={rf_metrics.rmse:.3f}\n\n"
        f"FORECAST (next {len(forecast)} business days):\n"
        + "\n".join(
            f"- {r.date}: blend={r.predicted_close_blend:.2f} (lr={r.predicted_close_lr:.2f}, rf={r.predicted_close_rf:.2f}), Δ%={r.expected_change_pct_from_last:+.2f}%"
            for r in forecast
        )
        + "\n\n"
        "TASK:\n"
        "- Explain the short-term trend (bullish/bearish/neutral) and why.\n"
        "- Suggest buy/sell/hold with risk management (entry ideas, stop-loss conceptually).\n"
        "- List top 3 risks seen in the data (volatility spike, volume drying up, reversal signs).\n"
    )


def find_previous_prediction(symbol: str, target_date: str, out_dir: str) -> Optional[float]:
    """
    Search existing JSON reports for the symbol to find a prediction for target_date.
    Returns the predicted_close_blend if found.
    """
    if not os.path.exists(out_dir):
        return None
    
    # List files matching SYMBOL_YYYYMMDD_HHMMSS.json
    import glob
    pattern = os.path.join(out_dir, f"{symbol.upper()}_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Skip if this report was generated ON the target_date (we want a PREVIOUS prediction)
                # though generated_at is ISO format, we can check the date part
                gen_date = data.get("generated_at", "").split("T")[0]
                if gen_date == target_date:
                    continue
                
                forecast = data.get("forecast", [])
                for row in forecast:
                    if row.get("date") == target_date:
                        return float(row.get("predicted_close_blend", 0.0))
        except Exception:
            continue
    return None


def save_report(
    out_dir: str,
    symbol: str,
    merged_csv_path: str,
    forecast: List[ForecastRow],
    lr_metrics: ModelMetrics,
    rf_metrics: ModelMetrics,
    trend: str,
    trend_reason: str,
    reco: str,
    reco_reason: str,
    ollama_text: Optional[str],
    plot_path: Optional[str],
    comparison: Optional[PredictionComparison] = None,
) -> Tuple[str, str]:
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{symbol.upper()}_{ts}"

    report_json_path = os.path.join(out_dir, f"{base}.json")
    report_txt_path = os.path.join(out_dir, f"{base}.txt")

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_csv": os.path.abspath(merged_csv_path),
        "plot_path": os.path.abspath(plot_path) if plot_path else None,
        "trend": {"label": trend, "reason": trend_reason},
        "recommendation": {"label": reco, "reason": reco_reason},
        "metrics": {"linear_regression": asdict(lr_metrics), "random_forest": asdict(rf_metrics)},
        "forecast": [asdict(r) for r in forecast],
        "comparison": asdict(comparison) if comparison else None,
        "ollama": {"model_output": ollama_text},
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append(f"NEPSE Prediction Report — {symbol.upper()}")
    lines.append(f"Generated: {payload['generated_at']}")
    lines.append(f"Dataset : {payload['dataset_csv']}")
    if plot_path:
        lines.append(f"Plot    : {payload['plot_path']}")
    lines.append("")
    lines.append(f"Trend: {trend.upper()} ({trend_reason})")
    lines.append(f"Recommendation: {reco.upper()} — {reco_reason}")
    lines.append("")
    lines.append("Model validation (time-based holdout):")
    lines.append(f"- Linear Regression: MAE={lr_metrics.mae:.3f}, RMSE={lr_metrics.rmse:.3f} (train={lr_metrics.n_train}, valid={lr_metrics.n_valid})")
    lines.append(f"- Random Forest    : MAE={rf_metrics.mae:.3f}, RMSE={rf_metrics.rmse:.3f} (train={rf_metrics.n_train}, valid={rf_metrics.n_valid})")
    lines.append("")
    lines.append(f"Forecast next {len(forecast)} business days (NPR):")
    for r in forecast:
        lines.append(f"- {r.date}: blend={r.predicted_close_blend:.2f} (lr={r.predicted_close_lr:.2f}, rf={r.predicted_close_rf:.2f}), Δ%={r.expected_change_pct_from_last:+.2f}%")
    lines.append("")

    if comparison:
        lines.append("Prediction Check (Accuracy for today):")
        lines.append(f"- Target Date    : {comparison.target_date}")
        lines.append(f"- Actual Close   : {comparison.actual_close:.2f} NPR")
        lines.append(f"- Predicted Close: {comparison.predicted_close:.2f} NPR")
        lines.append(f"- Difference     : {comparison.diff:+.2f} NPR")
        lines.append(f"- Error Rate     : {abs(comparison.error_pct):.2f}%")
        lines.append("")

    if ollama_text:
        lines.append("Ollama analysis:")
        lines.append(ollama_text.strip())
        lines.append("")
    else:
        lines.append("Ollama analysis: (not available — ollama not running or request failed)")
        lines.append("")

    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).strip() + "\n")

    return report_json_path, report_txt_path


def save_plot(out_dir: str, symbol: str, df: pd.DataFrame, forecast: List[ForecastRow]) -> Optional[str]:
    if not forecast:
        return None
    try:
        import matplotlib.pyplot as plt  # optional dependency
    except Exception:
        return None

    ensure_dir(out_dir)
    df = _standardize_ohlcv(df).sort_values(DATE_COL)
    hist = df.tail(120).copy()

    fut_dates = [pd.to_datetime(r.date) for r in forecast]
    fut_vals = [r.predicted_close_blend for r in forecast]

    plt.figure(figsize=(10, 5))
    plt.plot(hist[DATE_COL], hist["close"], label="Close (history)", linewidth=2)
    plt.plot(fut_dates, fut_vals, label="Forecast (blend)", linewidth=2)
    plt.axvline(hist[DATE_COL].iloc[-1], linestyle="--", alpha=0.5)
    plt.title(f"{symbol.upper()} — Close & {len(forecast)}-day Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price (NPR)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{symbol.upper()}_{ts}.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily NEPSE prediction system (data → features → models → forecast → report).")
    ap.add_argument("--symbol", required=True, help="NEPSE symbol (e.g. NABIL, NTC, UPPER)")
    ap.add_argument("--years", type=int, default=5, help="Years of history to fetch (default: 5)")
    ap.add_argument("--horizon", type=int, default=7, help="Forecast horizon (5–10 recommended, default: 7)")
    ap.add_argument("--data-dir", default="data", help="Directory to store growing CSV datasets")
    ap.add_argument("--out-dir", default="outputs", help="Directory to store reports/plots")
    ap.add_argument("--ollama-model", default="mistral", help="Ollama model name (default: mistral)")
    ap.add_argument("--no-ollama", action="store_true", help="Disable Ollama integration")
    ap.add_argument("--offline", action="store_true", help="Skip network fetch and use local CSV only (if available)")
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()
    horizon = int(np.clip(args.horizon, 1, 30))
    print(f"Running NEPSE system for {symbol} (horizon={horizon}, offline={bool(args.offline)})", flush=True)

    # Map symbol → company id where available (helps nepalstock fallback).
    cid = ""
    if not args.offline:
        try:
            companies = fetch_company_list()
            m = companies[companies["symbol"].str.upper() == symbol]
            if not m.empty:
                cid = str(m.iloc[0].get("id", "") or "")
        except Exception:
            cid = ""

    existing_path = os.path.join(args.data_dir, f"{symbol}.csv")
    existing = load_existing_csv(existing_path)

    # 1) Data collection (with robust fallback)
    fetched = None
    if not args.offline:
        try:
            fetched = fetch_history(symbol, company_id=cid, years=args.years)
        except Exception:
            fetched = None

    if fetched is None:
        # Fallback 1: use existing per-symbol dataset if present.
        if existing is not None and not existing.empty:
            merged = existing
        # Fallback 2: if project has old_data/new_data samples, merge them.
        elif os.path.exists("old_data.csv") and os.path.exists("new_data.csv"):
            old_df = _standardize_ohlcv(pd.read_csv("old_data.csv"))
            new_df = _standardize_ohlcv(pd.read_csv("new_data.csv"))
            merged = _standardize_ohlcv(pd.concat([old_df, new_df], ignore_index=True))
            merge_append_save(None, merged, existing_path)
        else:
            raise SystemExit(
                "Data fetch failed and no local dataset found.\n"
                "Try again with internet access, or provide an existing CSV at "
                f"'{existing_path}', or run with '--offline' after you have data."
            )
    else:
        merged = merge_append_save(existing, fetched, existing_path)

    if len(merged) < 80:
        raise SystemExit(f"Not enough data after cleaning ({len(merged)} rows). Need at least ~80 for ML features.")

    # 2) Data processing + feature engineering
    print(f"Dataset ready: {len(merged)} rows. Building features…", flush=True)
    X, y, feat = make_supervised(merged)
    X_train, X_valid, y_train, y_valid = time_split(X, y, valid_frac=0.2)

    feature_names = list(X.columns)
    models = build_models(feature_names)

    # 3) Model building
    print("Training Linear Regression + Random Forest (time-based holdout)…", flush=True)
    t = time.time()
    lr_metrics = fit_and_score(models["lr"], X_train, y_train, X_valid, y_valid)
    print(f"  LR done in {time.time() - t:.2f}s (MAE={lr_metrics.mae:.3f}, RMSE={lr_metrics.rmse:.3f})", flush=True)
    t = time.time()
    rf_metrics = fit_and_score(models["rf"], X_train, y_train, X_valid, y_valid)
    print(f"  RF done in {time.time() - t:.2f}s (MAE={rf_metrics.mae:.3f}, RMSE={rf_metrics.rmse:.3f})", flush=True)

    # Refit on full data for forecasting
    print("Refitting on full history and forecasting…", flush=True)
    models["lr"].fit(X, y)
    models["rf"].fit(X, y)

    # 4) Forecast
    forecast = recursive_forecast(merged, models["lr"], models["rf"], horizon=horizon)
    trend, trend_reason = classify_trend(merged, forecast)
    reco, reco_reason = rule_based_recommendation(merged, forecast)

    # 4.5) Compare with previous prediction if today is a market day
    comparison = None
    last_row = merged.iloc[-1]
    last_date_str = last_row[DATE_COL].strftime("%Y-%m-%d")
    
    # Simple check if today's data is fresh (within 24h of now)
    # or just assume if we have data for 'last_date_str', we should check for a prediction.
    prev_pred = find_previous_prediction(symbol, last_date_str, args.out_dir)
    if prev_pred is not None:
        actual = float(last_row["close"])
        diff = actual - prev_pred
        err = (diff / (prev_pred + 1e-9)) * 100.0
        comparison = PredictionComparison(
            target_date=last_date_str,
            actual_close=round(actual, 2),
            predicted_close=round(prev_pred, 2),
            diff=round(diff, 2),
            error_pct=round(err, 2),
        )

    # 5) Visualization
    plot_path = save_plot(args.out_dir, symbol, merged, forecast)

    # 6) Ollama integration
    ollama_text = None
    if not args.no_ollama:
        prompt = build_ollama_prompt(symbol, merged, forecast, lr_metrics, rf_metrics)
        ollama_text = call_ollama(prompt, model=args.ollama_model)

    # 7) Output + save
    report_json_path, report_txt_path = save_report(
        out_dir=args.out_dir,
        symbol=symbol,
        merged_csv_path=existing_path,
        forecast=forecast,
        lr_metrics=lr_metrics,
        rf_metrics=rf_metrics,
        trend=trend,
        trend_reason=trend_reason,
        reco=reco,
        reco_reason=reco_reason,
        ollama_text=ollama_text,
        plot_path=plot_path,
        comparison=comparison,
    )

    # Console output (high-signal)
    print(f"\nSymbol: {symbol}")
    print(f"Dataset CSV: {os.path.abspath(existing_path)} (rows={len(merged)})")
    if plot_path:
        print(f"Plot saved: {os.path.abspath(plot_path)}")
    print(f"Report saved: {os.path.abspath(report_txt_path)}")
    print(f"\nTrend: {trend.upper()} ({trend_reason})")
    print(f"Recommendation: {reco.upper()} — {reco_reason}")
    print("\nPredicted prices (blend):")
    for r in forecast:
        print(f"- {r.date}: {r.predicted_close_blend:.2f} NPR (Δ%={r.expected_change_pct_from_last:+.2f}%)")

    if comparison:
        print(f"\nPrediction Check for {comparison.target_date}:")
        print(f"  Actual: {comparison.actual_close:.2f} | Predicted: {comparison.predicted_close:.2f}")
        print(f"  Diff: {comparison.diff:+.2f} NPR ({comparison.error_pct:+.2f}%)")

    if ollama_text:
        print("\nOllama:")
        print(ollama_text.strip())

    # Also keep JSON path for integrations
    print(f"\nJSON: {os.path.abspath(report_json_path)}\n")


if __name__ == "__main__":
    main()

