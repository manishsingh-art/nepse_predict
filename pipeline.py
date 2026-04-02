from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest_engine import BacktestConfig, BacktestResult, generate_signals, run_backtest
from cache import dataframe_fingerprint, get_cache_path, load_pickle, save_pickle
from features import add_market_features, add_targets, build_features, clean_ohlcv_data, get_feature_cols
from models import NEPSEEnsemble, ForecastPoint, purged_walk_forward_splits
from regime import MarketRegimeDetector

try:
    from meta_labeling import MetaLabeler, MetaReport
    HAS_META = True
except ImportError:
    HAS_META = False
    MetaLabeler = None   # type: ignore[assignment,misc]
    MetaReport  = None   # type: ignore[assignment,misc]


@dataclass
class PipelineResult:
    clean_data: pd.DataFrame
    feature_frame: pd.DataFrame
    feature_cols: List[str]
    model: NEPSEEnsemble
    forecast: List[ForecastPoint]
    predictions: pd.DataFrame
    signals: pd.DataFrame
    backtest: BacktestResult
    meta_labeler: Optional[Any] = field(default=None)   # MetaLabeler instance
    meta_report: Optional[Any]  = field(default=None)   # MetaReport dataclass


def prepare_pipeline_frame(
    data: pd.DataFrame,
    market_data: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    clean_data = clean_ohlcv_data(data)
    if market_data is not None and not market_data.empty:
        clean_market = clean_ohlcv_data(market_data)
        clean_data = add_market_features(clean_data, clean_market)

    feature_frame = build_features(clean_data.copy(), include_context_features=False)
    feature_frame = add_targets(feature_frame)
    feature_frame = feature_frame.dropna(subset=["target_ret_1d", "target_date"]).reset_index(drop=True)
    feature_cols = get_feature_cols(feature_frame)
    if not feature_cols:
        raise ValueError("No usable feature columns after preparation.")
    return clean_data, feature_frame, feature_cols


def train_model(
    data: pd.DataFrame,
    symbol: str = "STOCK",
    market_data: Optional[pd.DataFrame] = None,
    optimise: bool = True,
    n_folds: Optional[int] = None,
    n_opt_trials: int = 10,
    random_state: int = 42,
) -> tuple[NEPSEEnsemble, pd.DataFrame, pd.DataFrame, List[str]]:
    cache_file = _model_cache_file(
        data=data,
        symbol=symbol,
        market_data=market_data,
        optimise=optimise,
        n_folds=n_folds,
        n_opt_trials=n_opt_trials,
        random_state=random_state,
    )
    cached_artifact = load_pickle(cache_file, max_age_seconds=None)
    if isinstance(cached_artifact, dict):
        model = cached_artifact.get("model")
        clean_data = cached_artifact.get("clean_data")
        feature_frame = cached_artifact.get("feature_frame")
        feature_cols = cached_artifact.get("feature_cols")
        if (
            isinstance(model, NEPSEEnsemble)
            and isinstance(clean_data, pd.DataFrame)
            and isinstance(feature_frame, pd.DataFrame)
            and isinstance(feature_cols, list)
        ):
            return model, clean_data, feature_frame, feature_cols

    clean_data, feature_frame, feature_cols = prepare_pipeline_frame(data, market_data=market_data)
    folds = n_folds if n_folds is not None else min(5, max(3, len(feature_frame) // 180))
    model = NEPSEEnsemble(
        symbol=symbol,
        n_folds=folds,
        optimise=optimise,
        n_opt_trials=n_opt_trials,
        random_state=random_state,
    )
    model.fit(feature_frame, feature_cols)
    save_pickle(
        cache_file,
        {
            "model": model,
            "clean_data": clean_data,
            "feature_frame": feature_frame,
            "feature_cols": feature_cols,
        },
    )
    return model, clean_data, feature_frame, feature_cols


def walk_forward_predictions(
    clean_data: pd.DataFrame,
    feature_frame: pd.DataFrame,
    feature_cols: List[str],
    symbol: str = "STOCK",
    optimise: bool = False,
    n_folds: Optional[int] = None,
    n_opt_trials: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    folds = n_folds if n_folds is not None else min(5, max(3, len(feature_frame) // 180))
    splits = purged_walk_forward_splits(
        len(feature_frame),
        n_folds=folds,
        min_train=max(60, len(feature_frame) // 5),
    )
    if not splits:
        splits = [(range(0, int(len(feature_frame) * 0.8)), range(int(len(feature_frame) * 0.8), len(feature_frame)))]

    regime_detector = MarketRegimeDetector()
    pred_rows: List[Dict[str, Any]] = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(splits, start=1):
        train_df = feature_frame.iloc[list(tr_idx)].copy().reset_index(drop=True)
        model = NEPSEEnsemble(
            symbol=symbol,
            n_folds=min(3, max(2, len(train_df) // 120)),
            optimise=optimise,
            n_opt_trials=n_opt_trials,
            random_state=random_state,
        )
        model.fit(train_df, feature_cols)

        for idx in list(vl_idx):
            row = feature_frame.iloc[[idx]].copy()
            row_data = row.iloc[0]
            signal_date = pd.Timestamp(row["date"].iloc[0])
            history = clean_data.loc[clean_data["date"] <= signal_date].copy()
            regime_info = regime_detector.detect_regime(history, None)

            point = model.predict_next_session(
                X_row=row,
                prev_price=float(row["close"].iloc[0]),
                history=history,
                next_date=pd.Timestamp(row["target_date"].iloc[0]).date(),
                regime_info=regime_info,
            )
            pred_rows.append(
                {
                    "fold": fold_idx,
                    "signal_date": signal_date.strftime("%Y-%m-%d"),
                    "trade_date": pd.Timestamp(row["target_date"].iloc[0]).strftime("%Y-%m-%d"),
                    "signal_close": float(row["close"].iloc[0]),
                    "target_close": float(row["target_next_close"].iloc[0]),
                    "predicted_close": float(point.predicted_close),
                    "predicted_return": float(point.predicted_return),
                    "actual_return": float(row["target_ret_1d"].iloc[0]),
                    "direction_prob": float(point.direction_prob),
                    "atr_pct_14": float(row_data.get("atr_pct_14", 0.0) or 0.0),
                    "vol_20d": float(row_data.get("vol_20d", 0.0) or 0.0),
                    "garch_vol": float(row_data.get("garch_vol", 0.0) or 0.0),
                    "volatility_20": float(row_data.get("volatility_20", 0.0) or 0.0),
                    "regime": str(regime_info.get("regime", row_data.get("regime", "NEUTRAL"))),
                    "regime_confidence": float(regime_info.get("confidence", 0.0) or 0.0),
                    "regime_volatility_pct": float(regime_info.get("volatility_pct", 0.0) or 0.0),
                }
            )

    if not pred_rows:
        return pd.DataFrame()
    return pd.DataFrame(pred_rows).sort_values("trade_date").reset_index(drop=True)


def fit_meta_labeler(
    predictions: pd.DataFrame,
    feature_frame: pd.DataFrame,
) -> "Optional[MetaLabeler]":
    """
    Train a MetaLabeler on the OOF predictions from walk_forward_predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Output of ``walk_forward_predictions``.
    feature_frame : pd.DataFrame
        Full feature frame including ``triple_barrier_label`` column.

    Returns
    -------
    MetaLabeler or None
        Fitted MetaLabeler when sklearn is available and enough samples exist.
        None when meta_labeling is unavailable or training fails.
    """
    if not HAS_META or predictions.empty:
        return None
    try:
        ml = MetaLabeler()
        ml.fit(predictions, feature_frame)
        return ml if ml.is_fitted_ else None
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("MetaLabeler.fit failed: %s", exc)
        return None


def run_full_pipeline(
    data: pd.DataFrame,
    symbol: str = "STOCK",
    market_data: Optional[pd.DataFrame] = None,
    forecast_horizon: int = 1,
    optimise: bool = True,
    n_folds: Optional[int] = None,
    n_opt_trials: int = 10,
    random_state: int = 42,
    backtest_config: Optional[BacktestConfig] = None,
    use_meta_labeling: bool = True,
) -> PipelineResult:
    """
    Run the full NEPSE prediction pipeline.

    Parameters
    ----------
    use_meta_labeling : bool
        When True (default), fits a MetaLabeler after the walk-forward
        predictions and passes ``meta_predictions`` to ``generate_signals``
        so that low-confidence trades are filtered.  Set False to reproduce
        the behaviour of the pipeline without meta-labeling (useful for
        baseline comparison).
    """
    model, clean_data, feature_frame, feature_cols = train_model(
        data=data,
        symbol=symbol,
        market_data=market_data,
        optimise=optimise,
        n_folds=n_folds,
        n_opt_trials=n_opt_trials,
        random_state=random_state,
    )
    predictions = walk_forward_predictions(
        clean_data=clean_data,
        feature_frame=feature_frame,
        feature_cols=feature_cols,
        symbol=symbol,
        optimise=False,
        n_folds=n_folds,
        n_opt_trials=n_opt_trials,
        random_state=random_state,
    )

    # ── Meta-Labeling (optional second-stage filter) ──────────────────────
    meta_labeler  = None
    meta_report   = None
    meta_preds_df = None

    if use_meta_labeling and HAS_META and not predictions.empty:
        meta_labeler = fit_meta_labeler(predictions, feature_frame)
        if meta_labeler is not None:
            meta_report   = meta_labeler.report_
            meta_preds_df = meta_labeler.predict_frame(predictions)

    signals = generate_signals(
        predictions,
        config=backtest_config,
        meta_predictions=meta_preds_df,
    )
    backtest = run_backtest(signals, market_data=clean_data, config=backtest_config)
    forecast = model.forecast(clean_data, feature_cols, horizon=forecast_horizon)

    return PipelineResult(
        clean_data=clean_data,
        feature_frame=feature_frame,
        feature_cols=feature_cols,
        model=model,
        forecast=forecast,
        predictions=predictions,
        signals=signals,
        backtest=backtest,
        meta_labeler=meta_labeler,
        meta_report=meta_report,
    )


def _model_cache_file(
    data: pd.DataFrame,
    symbol: str,
    market_data: Optional[pd.DataFrame],
    optimise: bool,
    n_folds: Optional[int],
    n_opt_trials: int,
    random_state: int,
):
    model_key = dataframe_fingerprint(data, ["date", "open", "high", "low", "close", "volume"])
    market_key = dataframe_fingerprint(market_data, ["date", "open", "high", "low", "close", "volume"])
    cache_token = "_".join(
        [
            symbol.upper(),
            f"d{model_key}",
            f"m{market_key}",
            f"opt{int(bool(optimise))}",
            f"f{n_folds or 0}",
            f"t{int(n_opt_trials)}",
            f"r{int(random_state)}",
        ]
    )
    return get_cache_path("models", f"{cache_token}.pkl")
