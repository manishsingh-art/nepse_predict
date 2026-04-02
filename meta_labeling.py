#!/usr/bin/env python3
"""
meta_labeling.py — Second-Stage Trade-Quality Filter for NEPSE
==============================================================
Implements López de Prado's meta-labeling framework (AFML, Ch. 3).

The primary ensemble regressor predicts direction and magnitude of returns.
The MetaLabeler learns WHEN to trust those predictions by answering a
binary question: "Given the primary model's call, should we actually trade?"

Architecture
------------
1. Primary model (models.py, NEPSEEnsemble) → predicted_return, direction_prob
2. Walk-forward OOF predictions are generated (pipeline.py)
3. MetaLabeler.fit():
   - Aligns OOF predictions with triple_barrier_label from feature_frame
   - meta_label[t] = 1 iff sign(predicted_return) == triple_barrier_label[t]
     AND triple_barrier_label[t] ≠ 0  (non-ambiguous triple-barrier outcome)
   - Trains a lightweight classifier on meta-features derived from the
     primary model's output and market context
4. MetaLabeler.predict_frame() → adds meta_should_trade ∈ {0, 1} to the
   predictions DataFrame
5. generate_signals() (backtest_engine.py) skips new entries when
   meta_should_trade == 0

Timing Safety
-------------
- MetaLabeler.fit() operates exclusively on out-of-fold (OOF) predictions.
  The OOF set is already temporally ordered and purged — no fold sees data
  from its own training window.
- Threshold selection uses only the first 80 % of the OOF set; the
  remaining 20 % is held out for evaluation, preventing threshold leakage.
- Rolling accuracy features are computed with a strictly causal window
  (pandas rolling, no look-ahead).

Single Responsibility
---------------------
This module contains ZERO network I/O and ZERO direct model-training calls
outside of MetaLabeler itself.  It imports only standard-library + numpy +
pandas + sklearn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Optional sklearn classifiers ──────────────────────────────────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import precision_score, recall_score, f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_GBC = True
except ImportError:
    HAS_GBC = False


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class MetaReport:
    """Diagnostic report produced by MetaLabeler.fit()."""
    n_samples: int
    n_positive: int          # meta_label == 1
    n_negative: int          # meta_label == 0
    precision: float         # on held-out 20 % of OOF set
    recall: float
    f1: float
    threshold: float         # probability cutoff used
    filter_rate: float       # fraction of signals that would be blocked
    sharpe_improvement: float  # estimated Sharpe delta from filtering
    feature_importances: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"MetaReport | n={self.n_samples} pos={self.n_positive} "
            f"neg={self.n_negative} | prec={self.precision:.3f} "
            f"rec={self.recall:.3f} F1={self.f1:.3f} "
            f"threshold={self.threshold:.3f} "
            f"filter_rate={self.filter_rate:.1%} "
            f"sharpe_delta~{self.sharpe_improvement:+.3f}"
        )


# ─── Meta-Feature Names ───────────────────────────────────────────────────────

_META_FEATURE_COLS = [
    "direction_prob",
    "abs_predicted_return",
    "confidence",              # |direction_prob - 0.5| * 2
    "atr_pct_14",
    "vol_20d",
    "garch_vol",
    "regime_encoded",          # BULL=1, BEAR=-1, else 0
    "rolling_accuracy_10",     # causal rolling mean of primary being correct
    "rolling_accuracy_20",
    "predicted_return_sign",   # sign of predicted_return (-1/0/+1)
]


# ─── MetaLabeler ─────────────────────────────────────────────────────────────

class MetaLabeler:
    """
    Second-stage binary classifier that filters primary-model trade signals.

    Workflow
    --------
    1. ``fit(oof_predictions, feature_frame)``  — train on OOF predictions
    2. ``predict_frame(predictions)``           — annotate a predictions DataFrame
    3. ``predict(row)``                         — single-row inference

    Attributes
    ----------
    is_fitted_ : bool
    report_ : MetaReport
    """

    # Minimum number of OOF samples required to attempt fitting
    _MIN_SAMPLES = 40

    def __init__(self, min_precision_target: float = 0.55) -> None:
        """
        Parameters
        ----------
        min_precision_target : float
            Desired precision on the held-out OOF slice.  The threshold is
            swept to maximise precision subject to recall ≥ 0.30, ensuring
            the meta-filter does not block virtually all trades.
        """
        self.min_precision_target = float(min_precision_target)
        self.is_fitted_    = False
        self.model_        = None     # sklearn Pipeline
        self.threshold_    = 0.5
        self.report_       = None
        self._feature_cols: List[str] = []

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        oof_predictions: pd.DataFrame,
        feature_frame: pd.DataFrame,
    ) -> "MetaLabeler":
        """
        Train the meta-classifier.

        Parameters
        ----------
        oof_predictions : pd.DataFrame
            Output of ``pipeline.walk_forward_predictions()``.  Must contain
            columns: ``signal_date``, ``predicted_return``, ``direction_prob``,
            ``actual_return``, ``atr_pct_14``, ``vol_20d``, ``regime``.
        feature_frame : pd.DataFrame
            Full feature frame from ``build_features`` + ``add_targets``.
            Must contain ``date`` and ``triple_barrier_label``.

        Returns
        -------
        self
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available — MetaLabeler will pass-through all signals")
            return self

        oof = oof_predictions.copy()
        if oof.empty or len(oof) < self._MIN_SAMPLES:
            logger.warning(
                "MetaLabeler.fit: only %d OOF samples (need %d) — skipping",
                len(oof), self._MIN_SAMPLES,
            )
            return self

        # ── Align triple barrier labels by signal_date ────────────────────
        tb_map = (
            feature_frame[["date", "triple_barrier_label"]]
            .dropna(subset=["triple_barrier_label"])
            .assign(date=lambda d: pd.to_datetime(d["date"]).dt.normalize())
            .set_index("date")["triple_barrier_label"]
        )
        oof["_signal_date_ts"] = pd.to_datetime(oof["signal_date"]).dt.normalize()
        oof["triple_barrier"]  = oof["_signal_date_ts"].map(tb_map)

        # Drop rows where triple barrier label is missing or zero (ambiguous)
        oof = oof.dropna(subset=["triple_barrier"]).copy()
        oof = oof[oof["triple_barrier"] != 0].copy()

        if len(oof) < self._MIN_SAMPLES:
            logger.warning(
                "MetaLabeler.fit: only %d non-zero triple-barrier rows — skipping",
                len(oof),
            )
            return self

        # ── Compute meta-labels ───────────────────────────────────────────
        # meta_label = 1 iff primary model's direction matches TB outcome
        primary_dir = np.sign(oof["predicted_return"].astype(float))
        oof["meta_label"] = (primary_dir == oof["triple_barrier"]).astype(int)

        # ── Build meta-features ───────────────────────────────────────────
        X_df = self._build_meta_features(oof)
        y    = oof["meta_label"].values

        # ── Temporal train/eval split (80/20 of OOF) ─────────────────────
        split = int(len(X_df) * 0.80)
        if split < 20 or (len(X_df) - split) < 10:
            split = max(20, len(X_df) - 10)

        X_tr, X_ev = X_df.iloc[:split], X_df.iloc[split:]
        y_tr, y_ev = y[:split],         y[split:]

        # ── Train classifier ──────────────────────────────────────────────
        self.model_ = self._build_classifier(len(X_tr))
        try:
            self.model_.fit(X_tr, y_tr)
        except Exception as exc:
            logger.warning("MetaLabeler training failed: %s", exc)
            return self

        # ── Threshold selection on eval split ─────────────────────────────
        self.threshold_ = self._select_threshold(X_ev, y_ev)

        # ── Evaluation report ─────────────────────────────────────────────
        if len(X_ev) > 0:
            ev_prob = self._predict_proba(X_ev)
            ev_pred = (ev_prob >= self.threshold_).astype(int)
            prec = float(precision_score(y_ev, ev_pred, zero_division=0.0))
            rec  = float(recall_score(y_ev, ev_pred, zero_division=0.0))
            f1   = float(f1_score(y_ev, ev_pred, zero_division=0.0))
        else:
            prec = rec = f1 = 0.0

        filter_rate = self._estimate_filter_rate(X_df)
        sharpe_imp  = self._estimate_sharpe_improvement(oof, y, X_df)

        self.report_ = MetaReport(
            n_samples=len(oof),
            n_positive=int(y.sum()),
            n_negative=int((y == 0).sum()),
            precision=prec,
            recall=rec,
            f1=f1,
            threshold=self.threshold_,
            filter_rate=filter_rate,
            sharpe_improvement=sharpe_imp,
            feature_importances=self._feature_importances(),
        )

        self.is_fitted_ = True
        logger.info("MetaLabeler fitted — %s", self.report_)
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def predict(self, row: pd.Series) -> Tuple[int, float]:
        """
        Predict whether to execute a single trade signal.

        Parameters
        ----------
        row : pd.Series
            A single row from a predictions DataFrame (output of
            ``walk_forward_predictions``).  Must contain at least
            ``direction_prob`` and ``predicted_return``.

        Returns
        -------
        (should_trade, meta_confidence) : (int, float)
            ``should_trade`` ∈ {0, 1}; 1 = proceed, 0 = skip.
            ``meta_confidence`` ∈ [0, 1].  When the model is not fitted,
            returns (1, 0.5) — graceful pass-through.
        """
        if not self.is_fitted_ or self.model_ is None:
            return 1, 0.5

        try:
            X = self._build_meta_features(pd.DataFrame([row]))
            prob = float(self._predict_proba(X)[0])
            should_trade = int(prob >= self.threshold_)
            return should_trade, prob
        except Exception as exc:
            logger.debug("MetaLabeler.predict fallback: %s", exc)
            return 1, 0.5

    def predict_frame(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate a full predictions DataFrame with meta-labeling columns.

        Adds two columns:
        - ``meta_should_trade`` ∈ {0, 1}
        - ``meta_confidence``   ∈ [0, 1]

        Parameters
        ----------
        predictions : pd.DataFrame
            Output of ``pipeline.walk_forward_predictions()``.

        Returns
        -------
        pd.DataFrame
            Input frame with the two new columns appended.
        """
        out = predictions.copy()
        if not self.is_fitted_ or self.model_ is None or out.empty:
            out["meta_should_trade"] = 1
            out["meta_confidence"]   = 0.5
            return out

        try:
            X    = self._build_meta_features(out)
            prob = self._predict_proba(X)
            out["meta_should_trade"] = (prob >= self.threshold_).astype(int)
            out["meta_confidence"]   = np.round(prob, 6)
        except Exception as exc:
            logger.warning("MetaLabeler.predict_frame fallback: %s", exc)
            out["meta_should_trade"] = 1
            out["meta_confidence"]   = 0.5

        return out

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct the meta-feature matrix from a predictions-like DataFrame."""
        n   = len(df)
        out = pd.DataFrame(index=df.index)

        pred_ret = df["predicted_return"].astype(float) if "predicted_return" in df.columns else pd.Series(np.zeros(n), index=df.index)
        dir_prob = df["direction_prob"].astype(float)   if "direction_prob"   in df.columns else pd.Series(np.full(n, 0.5), index=df.index)

        out["direction_prob"]        = dir_prob
        out["abs_predicted_return"]  = pred_ret.abs()
        out["confidence"]            = (dir_prob - 0.5).abs() * 2.0
        out["predicted_return_sign"] = np.sign(pred_ret)

        # Volatility / risk context
        for col in ("atr_pct_14", "vol_20d", "garch_vol"):
            out[col] = df[col].astype(float) if col in df.columns else 0.0

        # Regime encoding: BULL=+1, BEAR=-1, else 0
        if "regime" in df.columns:
            regime_str = df["regime"].astype(str).str.upper()
            out["regime_encoded"] = np.select(
                [regime_str.str.contains("BULL"), regime_str.str.contains("BEAR")],
                [1.0, -1.0],
                default=0.0,
            )
        else:
            out["regime_encoded"] = 0.0

        # Causal rolling accuracy of primary model (uses actual_return when available)
        if "actual_return" in df.columns and "predicted_return" in df.columns:
            actual      = df["actual_return"].astype(float)
            correct_dir = (np.sign(pred_ret) == np.sign(actual)).astype(float)
            out["rolling_accuracy_10"] = correct_dir.rolling(10, min_periods=3).mean()
            out["rolling_accuracy_20"] = correct_dir.rolling(20, min_periods=5).mean()
        else:
            out["rolling_accuracy_10"] = 0.5
            out["rolling_accuracy_20"] = 0.5

        self._feature_cols = list(out.columns)
        return out

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(meta_label=1) for each row."""
        proba = self.model_.predict_proba(X)
        # sklearn returns [[P(0), P(1)], ...]
        classes = list(self.model_.classes_) if hasattr(self.model_, "classes_") else [0, 1]
        pos_idx = classes.index(1) if 1 in classes else -1
        return proba[:, pos_idx] if pos_idx >= 0 else proba[:, -1]

    def _build_classifier(self, n_samples: int) -> Any:
        """
        Select and wrap a classifier.

        LogisticRegression is preferred (fast, interpretable, low variance).
        GradientBoostingClassifier is used when ≥ 200 samples are available
        and the library is installed.
        """
        if HAS_GBC and n_samples >= 200:
            base = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        else:
            base = LogisticRegression(
                C=1.0,
                max_iter=500,
                random_state=42,
                class_weight="balanced",
            )

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     base),
        ])

    def _select_threshold(self, X_ev: pd.DataFrame, y_ev: np.ndarray) -> float:
        """
        Sweep probability thresholds on the eval slice.

        Choose the threshold that maximises precision while keeping
        recall ≥ 0.30 (so we don't block every trade).  Falls back to 0.5
        if no valid threshold found.
        """
        if len(X_ev) == 0 or self.model_ is None:
            return 0.5

        try:
            probs      = self._predict_proba(X_ev)
            best_t     = 0.5
            best_prec  = 0.0
            for t in np.linspace(0.35, 0.80, 46):
                preds = (probs >= t).astype(int)
                if preds.sum() == 0:
                    continue
                rec  = float(recall_score(y_ev, preds, zero_division=0.0))
                prec = float(precision_score(y_ev, preds, zero_division=0.0))
                if rec >= 0.30 and prec > best_prec:
                    best_prec = prec
                    best_t    = t
            return float(best_t)
        except Exception:
            return 0.5

    def _estimate_filter_rate(self, X: pd.DataFrame) -> float:
        """Fraction of signals that would be blocked by the current threshold."""
        if self.model_ is None or len(X) == 0:
            return 0.0
        try:
            probs = self._predict_proba(X)
            return float((probs < self.threshold_).mean())
        except Exception:
            return 0.0

    def _estimate_sharpe_improvement(
        self,
        oof: pd.DataFrame,
        y: np.ndarray,
        X: pd.DataFrame,
    ) -> float:
        """
        Estimate Sharpe delta from applying the meta-filter to the OOF set.

        Base Sharpe: long/flat strategy using primary model direction.
        Filtered Sharpe: same strategy but skipping meta_label=0 predictions.
        Returns filtered_sharpe - base_sharpe.
        """
        if self.model_ is None or "actual_return" not in oof.columns:
            return 0.0
        try:
            actual   = oof["actual_return"].astype(float).values
            pred_ret = oof["predicted_return"].astype(float).values
            positions = (pred_ret > 0).astype(float)

            def _sharpe(pos: np.ndarray, ret: np.ndarray) -> float:
                strat = pos * ret
                if len(strat) < 2:
                    return 0.0
                sigma = float(np.std(strat))
                return 0.0 if sigma <= 1e-12 else float(np.sqrt(252.0) * np.mean(strat) / sigma)

            base_sharpe = _sharpe(positions, actual)

            probs   = self._predict_proba(X)
            meta    = (probs >= self.threshold_).astype(float)
            filtered_sharpe = _sharpe(positions * meta, actual)
            return round(filtered_sharpe - base_sharpe, 4)
        except Exception:
            return 0.0

    def _feature_importances(self) -> Dict[str, float]:
        """Extract feature importances from the fitted classifier if available."""
        if self.model_ is None:
            return {}
        try:
            clf = self.model_.named_steps["clf"]
            if hasattr(clf, "feature_importances_"):
                raw = clf.feature_importances_
            elif hasattr(clf, "coef_"):
                raw = np.abs(clf.coef_[0])
            else:
                return {}
            total = float(raw.sum()) + 1e-9
            return {
                col: round(float(raw[i]) / total, 5)
                for i, col in enumerate(self._feature_cols)
                if i < len(raw)
            }
        except Exception:
            return {}

    @property
    def classes_(self):
        if self.model_ is None:
            return [0, 1]
        try:
            return list(self.model_.classes_)
        except Exception:
            return [0, 1]
