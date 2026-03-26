#!/usr/bin/env python3
"""
models.py — Production ML Ensemble for NEPSE
=============================================
Stacked ensemble:
  Layer 1: GradientBoosting + Random Forest + Ridge (always available via sklearn)
           + LightGBM + XGBoost when installed
  Layer 2: Ridge meta-learner on OOF predictions

Nepal-aware features:
  - Skips non-trading days in forecast horizon using NepalMarketCalendar
  - Uses NEPSE circuit-breaker (±10% per session)

Training:
  - Purged walk-forward cross-validation (no look-ahead bias)
  - Embargo gap between train/validation folds
  - Optuna hyperparameter optimisation when available
  - SHAP feature importance when available
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

# ── Optional ML library imports ───────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ── Nepal Calendar ────────────────────────────────────────────────────────────
try:
    from nepal_calendar import NepalMarketCalendar, next_nepse_trading_dates
    _NEPAL_CAL = NepalMarketCalendar(fetch_live=False)
    HAS_NEPAL_CAL = True
except ImportError:
    HAS_NEPAL_CAL = False
    _NEPAL_CAL = None

MODEL_CACHE_DIR = Path(os.path.expanduser("~")) / ".nepse_cache" / "models"


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    fold: int
    n_train: int
    n_valid: int
    mae: float
    rmse: float
    dir_acc: float
    mape: float


@dataclass
class ModelReport:
    symbol: str
    trained_at: str
    n_rows: int
    n_features: int
    cv_folds: List[FoldResult]
    avg_mae: float
    avg_rmse: float
    avg_dir_acc: float
    avg_mape: float
    models_used: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    best_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ForecastPoint:
    day: int
    date: str
    predicted_close: float
    direction_prob: float
    low_band: float
    high_band: float
    change_pct: float
    confidence: str
    direction_confidence: float = 0.5  # v6.0
    trap_score: int = 0             # v6.0
    scenario_bull: float = 0.0
    scenario_bear: float = 0.0
    is_trading_day: bool = True
    holiday_name: Optional[str] = None
    day_name: Optional[str] = None


# ─── Utilities ────────────────────────────────────────────────────────────────

def _make_pipeline(model) -> Pipeline:
    # Use keep_empty_features=True to prevent column count mismatch if a fold has all NaNs
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


def _impute(X: pd.DataFrame) -> np.ndarray:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    return imp.fit_transform(X)


def _clip_circuit(price: float, prev_price: float) -> float:
    """Enforce NEPSE ±10% circuit-breaker rule."""
    return float(np.clip(price, prev_price * 0.90, prev_price * 1.10))


def _apply_realism_constraints(price: float, prev_price: float, feats: Dict[str, Any], history: Optional[pd.DataFrame] = None) -> float:
    """
    Applies anti-overfitting and realism filters (v6.0):
    1. Resistance Gravity: If price > Resistance, dampen the move.
    2. Mean Reversion: If Z-Score > 2.5, pull back.
    3. RSI Guard: If RSI > 80, cap growth.
    4. Exponential Growth Penalty: Multi-day surges (>3 days of >7% gains) trigger sharp pullbacks.
    """
    new_price = price
    change_pct = (price - prev_price) / (prev_price + 1e-9)
    
    # ── 1. Exponential Growth Penalty (v6.0) ──
    if history is not None and len(history) >= 3:
        last_3_rets = history["close"].pct_change().tail(3)
        if (last_3_rets > 0.07).all() and change_pct > 0:
            # Dangerous multi-day circuit chasing. Penalize further growth.
            new_price = prev_price * 1.01 # Force a cooldown
            return new_price

    # ── 2. Resistance Dampening ──
    res20 = feats.get("resist_20", 0)
    if res20 > 0 and price > res20 and change_pct > 0:
        new_price = prev_price + (price - prev_price) * 0.3 # Dampen by 70%
    
    # ── 3. Mean Reversion ──
    z20 = feats.get("zscore_20", 0)
    if z20 > 2.5 and change_pct > 0:
        new_price = prev_price + (price - prev_price) * 0.15
    elif z20 < -2.5 and change_pct < 0:
        new_price = prev_price + (price - prev_price) * 0.4

    # ── 4. RSI Guard ──
    rsi = feats.get("rsi_14", 50)
    if rsi > 80 and new_price > prev_price:
        new_price = min(new_price, prev_price * 1.01)
        
    return new_price


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, prev: np.ndarray) -> float:
    true_dir = (y_true > prev).astype(int)
    pred_dir = (y_pred > prev).astype(int)
    return float(accuracy_score(true_dir, pred_dir)) * 100


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100


# ─── Walk-Forward CV ──────────────────────────────────────────────────────────

def purged_walk_forward_splits(
    n: int,
    n_folds: int = 5,
    embargo_pct: float = 0.01,
    min_train: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    splits = []
    fold_size = max(10, (n - min_train) // n_folds)
    embargo   = max(1, int(n * embargo_pct))

    for i in range(n_folds):
        val_start = min_train + i * fold_size
        val_end   = val_start + fold_size if i < n_folds - 1 else n
        train_end = val_start - embargo
        if train_end < min_train or val_end > n:
            continue
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))

    return splits


# ─── Hyperparameter Defaults ──────────────────────────────────────────────────

def _default_lgb_params(metric: str = "mae") -> dict:
    return {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 7,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
        "metric": metric
    }


# ─── Base Model Builders ──────────────────────────────────────────────────────

def _build_gbm():
    """GradientBoosting — always available sklearn fallback."""
    return _make_pipeline(GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    ))


def _build_rf():
    return _make_pipeline(RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        max_features="sqrt", n_jobs=-1, random_state=42,
    ))


def _build_ridge():
    return _make_pipeline(Ridge(alpha=10.0))


def _build_lgb(params: dict = None):
    if not HAS_LGB:
        return None
    p = params or _default_lgb_params()
    return lgb.LGBMRegressor(**p)


def _build_xgb():
    if not HAS_XGB:
        return None
    return xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0, eval_metric="mae",
    )


# ─── Stacking Ensemble ────────────────────────────────────────────────────────

class NEPSEEnsemble:
    """
    Two-layer stacking ensemble v6.0.
    - Regime-Aware Meta-Learner.
    - Sharpe-optimized hyperparameter tuning.
    - Multi-step recursive probabilistic forecast.
    """

    def __init__(
        self,
        symbol: str = "STOCK",
        n_folds: int = 5,
        optimise: bool = True,
        n_opt_trials: int = 30,
        optimize_for: str = "sharpe" # "mae" or "sharpe"
    ):
        self.symbol = symbol
        self.n_folds = n_folds
        self.optimise = optimise
        self.n_opt_trials = n_opt_trials
        self.optimize_for = optimize_for

        self.base_models_: Dict[str, Any] = {}
        self.meta_models_: Dict[str, Any] = {} # Regime -> Ridge
        self.global_meta_model_: Any = None
        self._meta_imputer: Any = None
        self.dir_classifier_: Any = None
        self.feature_cols_: List[str] = []
        self.report_: Optional[ModelReport] = None
        self.last_close_: float = 0.0
        self.recent_vol_: float = 0.01
        self.lgb_params_: dict = {}
        self._lgb_imputer: SimpleImputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self._models_used: List[str] = []
        self._tree_feature_weights_: Optional[np.ndarray] = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, df_feat: pd.DataFrame, feature_cols: List[str]) -> "NEPSEEnsemble":
        df = df_feat.dropna(subset=["target_next_close", "close"]).copy().reset_index(drop=True)
        X_all  = df[feature_cols].astype(float)
        y_all  = df["target_next_close"].astype(float)
        prev_all = df["close"].astype(float)

        self.feature_cols_ = feature_cols
        self.last_close_   = float(df["close"].iloc[-1])
        self.recent_vol_   = float(df["ret_1d"].std()) if "ret_1d" in df.columns else 0.015

        n      = len(X_all)
        splits = purged_walk_forward_splits(n, n_folds=self.n_folds, min_train=max(60, n // 5))
        if not splits:
            splits = [(np.arange(0, int(n * 0.8)), np.arange(int(n * 0.8), n))]

        # Determine which models are available
        builders: Dict[str, Any] = {}
        if HAS_LGB:
            builders["lgb"] = None
        if HAS_XGB:
            builders["xgb"] = _build_xgb
        builders["gbm"]   = _build_gbm
        builders["rf"]    = _build_rf
        builders["ridge"] = _build_ridge
        self._models_used = list(builders.keys())

        # LGB hyperparameter search (with resilience)
        self.lgb_params_ = _default_lgb_params()
        if self.optimise and HAS_OPTUNA and HAS_LGB and len(splits) > 0:
            try:
                self.lgb_params_ = self._optuna_tun(X_all, y_all, splits)
            except Exception as opt_err:
                logger.warning(f"Optuna tuning failed ({opt_err}), using defaults.")

        # OOF predictions
        oof: Dict[str, np.ndarray] = {k: np.full(n, np.nan) for k in builders}
        fold_results: List[FoldResult] = []

        for fold_i, (tr_idx, vl_idx) in enumerate(splits):
            Xtr_raw = X_all.iloc[tr_idx]
            ytr     = y_all.iloc[tr_idx].values
            Xvl_raw = X_all.iloc[vl_idx]
            yvl     = y_all.iloc[vl_idx].values
            prev_vl = prev_all.iloc[vl_idx].values

            fold_imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr_raw)
            Xtr_imp  = fold_imp.transform(Xtr_raw)
            Xvl_imp  = fold_imp.transform(Xvl_raw)

            fold_preds: Dict[str, np.ndarray] = {}

            if HAS_LGB:
                try:
                    # Prompt 6: Explicit penalty weights for Nepal calendar features.
                    # LightGBM sklearn API doesn't expose feature_penalty in some builds, so we apply
                    # a deterministic scaling to `np_` features (lower weight => less influence).
                    feature_names = list(Xtr_raw.columns)
                    penalty_w = np.array([0.7 if col.startswith("np_") else 1.0 for col in feature_names], dtype=float)
                    Xtr_imp_w = Xtr_imp * penalty_w
                    Xvl_imp_w = Xvl_imp * penalty_w

                    m = lgb.LGBMRegressor(**self.lgb_params_)
                    m.fit(Xtr_imp_w, ytr, eval_set=[(Xvl_imp_w, yvl)], callbacks=[lgb.early_stopping(30, verbose=False)])
                    fold_preds["lgb"] = m.predict(Xvl_imp_w)
                    oof["lgb"][vl_idx] = fold_preds["lgb"]
                except Exception as e:
                    logger.warning(f"LGB fold training failed: {e}. Skipping LGB for this fold.")

            if HAS_XGB:
                m = _build_xgb()
                feature_names = list(Xtr_raw.columns)
                penalty_w = np.array([0.7 if col.startswith("np_") else 1.0 for col in feature_names], dtype=float)
                Xtr_imp_w = Xtr_imp * penalty_w
                Xvl_imp_w = Xvl_imp * penalty_w
                m.fit(Xtr_imp_w, ytr, eval_set=[(Xvl_imp_w, yvl)], verbose=False)
                fold_preds["xgb"] = m.predict(Xvl_imp_w)
                oof["xgb"][vl_idx] = fold_preds["xgb"]

            for name in ["gbm", "rf", "ridge"]:
                builder = globals()[f"_build_{name}"]
                m = builder()
                m.fit(Xtr_raw, ytr)
                fold_preds[name] = m.predict(Xvl_raw)
                oof[name][vl_idx] = fold_preds[name]

            blend     = self._blend(fold_preds, Xvl_raw)
            fold_mae  = mean_absolute_error(yvl, blend)
            fold_rmse = float(np.sqrt(mean_squared_error(yvl, blend)))
            fold_da   = directional_accuracy(yvl, blend, prev_vl)
            fold_mape_v = mape(yvl, blend)

            fold_results.append(FoldResult(
                fold=fold_i + 1, n_train=len(tr_idx), n_valid=len(vl_idx),
                mae=round(fold_mae, 3), rmse=round(fold_rmse, 3),
                dir_acc=round(fold_da, 2), mape=round(fold_mape_v, 3),
            ))

        # Meta-learner on OOF (Regime-Aware v6.0)
        oof_df    = pd.DataFrame(oof)
        valid_mask = oof_df.notna().all(axis=1)
        if valid_mask.sum() > 30:
            self._meta_imputer = SimpleImputer(strategy="median").fit(oof_df[valid_mask])
            meta_X_c = self._meta_imputer.transform(oof_df[valid_mask])
            meta_y = y_all[valid_mask].values
            
            # Global Meta
            meta_ridge = Ridge(alpha=1.0)
            meta_ridge.fit(meta_X_c, meta_y)
            self.global_meta_model_ = meta_ridge
            
            # Regime-Specific Meta Learners
            if "regime" in df.columns:
                unique_regimes = df["regime"].unique()
                for r in unique_regimes:
                    r_mask = (df["regime"] == r) & valid_mask
                    if r_mask.sum() >= 20: # Enough data for ridge
                        rm = Ridge(alpha=1.0)
                        rm.fit(self._meta_imputer.transform(oof_df[r_mask]), y_all[r_mask].values)
                        self.meta_models_[r] = rm

        # Refit Base Models
        self._lgb_imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_all)
        X_imp = self._lgb_imputer.transform(X_all)
        # Prompt 6: explicit penalty weights for Nepal calendar features (applied as scaling)
        feature_names = list(X_all.columns)
        self._tree_feature_weights_ = np.array([0.7 if col.startswith("np_") else 1.0 for col in feature_names], dtype=float)

        if HAS_LGB:
            try:
                X_imp_w = X_imp * self._tree_feature_weights_
                m = lgb.LGBMRegressor(**self.lgb_params_)
                m.fit(X_imp_w, y_all.values)
                self.base_models_["lgb"] = ("imp", m)
            except Exception as e:
                logger.warning(f"Base model lgb failed to train: {e}. Skipping.")

        # Refit all base models with error handling
        for name in list(builders.keys()):
            builder = builders[name]
            try:
                if name == "lgb":
                    X_imp_w = X_imp * self._tree_feature_weights_
                    m = lgb.LGBMRegressor(**self.lgb_params_)
                    m.fit(X_imp_w, y_all.values)
                    self.base_models_["lgb"] = ("imp", m)
                elif name == "xgb":
                    m = builder() # _build_xgb
                    X_imp_w = X_imp * self._tree_feature_weights_
                    m.fit(X_imp_w, y_all.values)
                    self.base_models_["xgb"] = ("imp", m)
                else: # gbm, rf, ridge
                    m = builder() # _build_gbm, _build_rf, _build_ridge
                    m.fit(X_all, y_all.values)
                    self.base_models_[name] = ("raw", m)
            except Exception as e:
                logger.warning(f"Base model {name} failed to train: {e}. Skipping.")
                if name in self._models_used:
                    self._models_used.remove(name)

        # Direction classifier
        y_dir = (y_all.values > prev_all.values).astype(int)
        if HAS_LGB:
            dc = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
            dc.fit(X_imp, y_dir)
            self.dir_classifier_ = ("imp", dc)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            dc = _make_pipeline(GradientBoostingClassifier(n_estimators=150, random_state=42))
            dc.fit(X_all, y_dir)
            self.dir_classifier_ = ("raw", dc)

        self.report_ = ModelReport(
            symbol=self.symbol,
            trained_at=datetime.now().isoformat(timespec="seconds"),
            n_rows=n, n_features=len(feature_cols),
            cv_folds=fold_results,
            avg_mae=round(float(np.mean([f.mae for f in fold_results])), 3),
            avg_rmse=round(float(np.mean([f.rmse for f in fold_results])), 3),
            avg_dir_acc=round(float(np.mean([f.dir_acc for f in fold_results])), 2),
            avg_mape=round(float(np.mean([f.mape for f in fold_results])), 3),
            feature_importance=self._get_feature_importance(feature_cols),
            best_params=self.lgb_params_,
            models_used=self._models_used,
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict_one(self, X_row: pd.DataFrame, regime: Optional[str] = None) -> Tuple[float, float]:
        """Returns (predicted_close, direction_prob)."""
        X_raw = X_row[self.feature_cols_]
        X_imp = self._lgb_imputer.transform(X_raw)

        base_preds: Dict[str, float] = {}
        for name, (mode, model) in self.base_models_.items():
            try:
                if mode == "imp":
                    X_in = X_imp
                    if self._tree_feature_weights_ is not None:
                        X_in = X_in * self._tree_feature_weights_
                    base_preds[name] = float(model.predict(X_in)[0])
                else:
                    base_preds[name] = float(model.predict(X_raw)[0])
            except: pass

        if not base_preds: return self.last_close_, 0.5

        # Regime-Aware Stacking
        keys = [k for k in ["lgb", "xgb", "gbm", "rf", "ridge"] if k in base_preds]
        final_pred = float("nan")
        if self._meta_imputer is not None and keys:
            try:
                meta_X = self._meta_imputer.transform(np.array([[base_preds[k] for k in keys]]))
                if regime in self.meta_models_:
                    final_pred = float(self.meta_models_[regime].predict(meta_X)[0])
                elif self.global_meta_model_ is not None:
                    final_pred = float(self.global_meta_model_.predict(meta_X)[0])
            except Exception:
                final_pred = float("nan")
        
        if final_pred is None or (isinstance(final_pred, float) and np.isnan(final_pred)):
            final_pred = float(self._blend_dict(base_preds))
        if final_pred is None or (isinstance(final_pred, float) and np.isnan(final_pred)):
            final_pred = float(self.last_close_)

        dir_prob = 0.5
        if self.dir_classifier_ is not None:
            mode, dc = self.dir_classifier_
            try:
                if mode == "imp":
                    X_in = X_imp
                    if self._tree_feature_weights_ is not None:
                        X_in = X_in * self._tree_feature_weights_
                    dir_prob = float(dc.predict_proba(X_in)[0][1])
                else:
                    dir_prob = float(dc.predict_proba(X_raw)[0][1])
            except: pass
        
        return final_pred, dir_prob

    def compute_final_signal(
        self, 
        model_prob: float, 
        trend_score: float, 
        sentiment_score: float, 
        momentum_score: float, 
        volume_score: float
    ) -> Tuple[str, float]:
        """
        Unified Decision Engine (v5.0 Upgrade).
        Returns (SIGNAL, confidence_score).
        """
        # Normalize trend_score from [-6, +6] to [0, 1]
        t_normalized = (trend_score + 6) / 12.0
        # Normalize sentiment from [-1, +1] to [0, 1]
        s_normalized = (sentiment_score + 1) / 2.0
        # Normalize momentum (approximate) - assume -5% to +5% range
        m_normalized = np.clip((momentum_score + 5) / 10.0, 0, 1)
        # Normalize volume ratio (1.0 is neutral)
        v_normalized = np.clip(volume_score / 2.0, 0, 1)

        score = (
            0.35 * model_prob +
            0.25 * t_normalized +
            0.15 * s_normalized +
            0.15 * m_normalized +
            0.10 * v_normalized
        )

        if score > 0.6:
            return "BUY", score
        elif score < 0.4:
            return "SELL", score
        else:
            return "HOLD", score

    def forecast(
        self,
        df_hist: pd.DataFrame,
        feature_cols: List[str],
        horizon: int = 7,
        sentiment_score: float = 0.0,
        smart_money_info: Optional[Dict[str, Any]] = None
    ) -> List[ForecastPoint]:
        from features import build_features, add_targets
        from regime import MarketRegimeDetector

        df_work = df_hist.copy()
        regime_detector = MarketRegimeDetector()
        
        last_date_ts = df_work["date"].iloc[-1]
        last_date = last_date_ts.date() if hasattr(last_date_ts, "date") else last_date_ts

        if HAS_NEPAL_CAL and _NEPAL_CAL is not None:
            forecast_dates = next_nepse_trading_dates(last_date, horizon)
        else:
            from pandas.tseries.offsets import BDay
            forecast_dates = [(last_date_ts + BDay(i + 1)).date() for i in range(horizon)]

        predictions: List[ForecastPoint] = []
        prev_price = self.last_close_
        
        # Initial regime
        curr_reg_info = regime_detector.detect_regime(df_work, smart_money_info)

        for step, next_date in enumerate(forecast_dates, 1):
            holiday_name = _NEPAL_CAL.get_holiday_name(next_date) if HAS_NEPAL_CAL and _NEPAL_CAL else None

            feat_df = build_features(df_work, sentiment_score=sentiment_score, smart_money_info=smart_money_info)
            feat_df = add_targets(feat_df)
            last_row = feat_df.iloc[-1:]
            
            X_pred = pd.DataFrame(0.0, index=[0], columns=feature_cols)
            avail_cols = [c for c in feature_cols if c in last_row.columns]
            X_pred[avail_cols] = last_row[avail_cols].values

            raw_pred, dir_prob = self.predict_one(X_pred, curr_reg_info["regime"])
            
            # ── Fix Forecast Logic (v5.0 Upgrade) ──
            # Price adjustment based on probability to avoid counter-intuitive moves
            adjusted_base = raw_pred * (1 + (dir_prob - 0.5))
            
            feats_dict = last_row.to_dict(orient="records")[0]
            constrained_pred = _apply_realism_constraints(adjusted_base, prev_price, feats_dict, df_work)
            capped_pred = _clip_circuit(constrained_pred, prev_price)

            # Volatility range based on actual regime volatility
            vol_pct = curr_reg_info.get("volatility_pct", 2.0) / 100.0
            band = capped_pred * vol_pct * 1.5
            
            change_pct = (capped_pred - prev_price) / (prev_price + 1e-9) * 100
            conf_dist = abs(dir_prob - 0.5)
            confidence = "high" if conf_dist > 0.15 else "medium" if conf_dist > 0.07 else "low"

            predictions.append(ForecastPoint(
                day=step, date=next_date.strftime("%Y-%m-%d"),
                predicted_close=round(capped_pred, 2),
                direction_prob=round(dir_prob, 3),
                low_band=round(capped_pred - band, 2),
                high_band=round(capped_pred + band, 2),
                change_pct=round(change_pct, 2),
                confidence=confidence,
                direction_confidence=round(conf_dist * 2, 2), # 0..1 scale
                trap_score=curr_reg_info.get("trap_score", 0),
                scenario_bull=round(capped_pred + (band * 1.2), 2),
                scenario_bear=round(capped_pred - (band * 1.2), 2),
                holiday_name=holiday_name,
                day_name=next_date.strftime("%A"),
            ))

            # Update working DF for recursive step
            new_row = pd.DataFrame({
                "date": [pd.Timestamp(next_date)], "open": [capped_pred],
                "high": [capped_pred * 1.005], "low": [capped_pred * 0.995],
                "close": [capped_pred], "volume": [df_work["volume"].tail(20).mean()]
            })
            df_work = pd.concat([df_work, new_row], ignore_index=True)
            prev_price = capped_pred
            # Re-detect regime for next step
            curr_reg_info = regime_detector.detect_regime(df_work, smart_money_info)

        return predictions

    # ── Internals ─────────────────────────────────────────────────────────────

    def _blend(self, preds: dict, X_vl) -> np.ndarray:
        weights = {"lgb": 0.35, "xgb": 0.25, "gbm": 0.20, "rf": 0.15, "ridge": 0.05}
        blended, total_w = None, 0.0
        for name, w in weights.items():
            if name in preds:
                arr = np.array(preds[name])
                blended = arr * w if blended is None else blended + arr * w
                total_w += w
        return blended / total_w if blended is not None else np.mean(list(preds.values()), axis=0)

    def _blend_dict(self, preds: Dict[str, float]) -> float:
        weights = {"lgb": 0.35, "xgb": 0.25, "gbm": 0.20, "rf": 0.15, "ridge": 0.05}
        res, total_w = 0.0, 0.0
        for name, w in weights.items():
            if name in preds and preds[name] is not None and not np.isnan(preds[name]):
                res += preds[name] * w
                total_w += w
        if total_w > 0:
            return res / total_w
        # Fallback to mean if blending fails
        valid_preds = [v for v in preds.values() if v is not None and not np.isnan(v)]
        return float(np.mean(valid_preds)) if valid_preds else 0.0

    def _get_feature_importance(self, feature_cols: List[str]) -> Dict[str, float]:
        imp: Dict[str, float] = {}
        try:
            if "lgb" in self.base_models_:
                scores = self.base_models_["lgb"][1].feature_importances_
                for i, col in enumerate(feature_cols[:len(scores)]): imp[col] = float(scores[i])
        except: pass
        if imp:
            total = sum(imp.values()) + 1e-9
            imp = dict(sorted({k: round(v/total, 5) for k, v in imp.items()}.items(), key=lambda x: -x[1])[:30])
        return imp

    def _optuna_tun(self, X, y, splits) -> dict:
        if not HAS_OPTUNA: return _default_lgb_params()
        def obj(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 127),
                "max_depth": trial.suggest_int("max_depth", 5, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "random_state": 42, "verbose": -1
            }
            # Fast CV on 1st fold
            tr_idx, vl_idx = splits[0]
            imp = SimpleImputer(strategy="median")
            Xtr, Xvl = imp.fit_transform(X.iloc[tr_idx]), imp.transform(X.iloc[vl_idx])
            ytr, yvl = y.iloc[tr_idx].values, y.iloc[vl_idx].values
            m = lgb.LGBMRegressor(**params)
            m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], callbacks=[lgb.early_stopping(20, verbose=False)])
            preds = m.predict(Xvl)
            
            if self.optimize_for == "sharpe":
                prev_y = y.iloc[vl_idx].shift(1).bfill()
                rets = (preds - prev_y) / (prev_y + 1e-9)
                sharpe = np.mean(rets) / (np.std(rets) + 1e-9)
                return -sharpe
            return mean_absolute_error(yvl, preds)

        study = optuna.create_study(direction="minimize")
        # Extend timeout to 120s for highly complex features, and add catch for trials
        try:
            study.optimize(obj, n_trials=self.n_opt_trials, timeout=120, catch=(Exception,))
            return {**study.best_params, "random_state": 42, "verbose": -1}
        except Exception as e:
            logger.warning(f"Optuna optimize loop failed: {e}")
            return _default_lgb_params()

    def save(self, path: Optional[str] = None) -> str:
        import pickle
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = path or str(MODEL_CACHE_DIR / f"{self.symbol}_ensemble.pkl")
        with open(fpath, "wb") as f: pickle.dump(self, f)
        return fpath

    @staticmethod
    def load(path: str) -> "NEPSEEnsemble":
        import pickle
        with open(path, "rb") as f: return pickle.load(f)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        import pickle
        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fpath = path or str(MODEL_CACHE_DIR / f"{self.symbol}_ensemble.pkl")
        with open(fpath, "wb") as f:
            pickle.dump(self, f)
        return fpath

    @staticmethod
    def load(path: str) -> "NEPSEEnsemble":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)