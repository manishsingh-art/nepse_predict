import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
import os
import sys

# Ensure parent directory is in path to import original modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from fetcher import fetch_history, fetch_company_list, fetch_news_sentiment, get_aggregate_sentiment, fetch_live_price
from features import build_features, add_targets, get_feature_cols
from models import NEPSEEnsemble
from smart_money import SmartMoneyAnalyst
from strategy import TradingStrategyEngine
from analyze import add_indicators, detect_trend, detect_anomalies
from regime import MarketRegimeDetector

logger = logging.getLogger(__name__)

def standardize(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return standardize(obj.tolist())
    if isinstance(obj, dict):
        return {k: standardize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [standardize(i) for i in obj]
    return obj

def serialize_forecast(ml_preds):
    """Serialize forecast objects to dicts with all fields."""
    result = []
    for f in ml_preds:
        row = {}
        # Handle both dict-like and object-like forecast entries
        if hasattr(f, '__dict__'):
            row = {k: v for k, v in f.__dict__.items() if not k.startswith('_')}
        elif isinstance(f, dict):
            row = dict(f)
        else:
            try:
                row = dict(f)
            except:
                row = {"predicted_close": getattr(f, "predicted_close", 0)}
        result.append(row)
    return standardize(result)

class PredictorService:
    def __init__(self):
        self.sm_analyst = SmartMoneyAnalyst()
        self.strat_engine = TradingStrategyEngine()
        self.ensemble = NEPSEEnsemble()
        self.regime_detector = MarketRegimeDetector()

    def run_prediction(self, symbol: str, years: int = 2, n_predict: int = 7) -> Dict[str, Any]:
        """
        Runs the full ML pipeline for a given symbol and returns ALL structured data
        matching the nepse_live.py terminal output.
        """
        try:
            # 1. Fetch Data
            df = fetch_history(symbol, years=years)
            if df.empty:
                raise ValueError(f"No history found for {symbol}")

            # Patch live price if available
            live_price = fetch_live_price(symbol)
            if live_price and live_price > 0:
                df.loc[df.index[-1], "close"] = live_price

            # Dataset summary metrics
            last_close = float(df["close"].iloc[-1])
            prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else last_close
            week52_high = float(df["close"].tail(252).max())
            week52_low = float(df["close"].tail(252).min())
            avg_vol_20d = float(df["volume"].tail(20).mean()) if "volume" in df.columns else 0.0
            date_range_start = str(df.index[0].date()) if hasattr(df.index[0], 'date') else str(df.index[0])
            date_range_end = str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1])
            records_count = len(df)

            # 2. Technical Analysis
            df = add_indicators(df)
            trend_info = detect_trend(df)
            anomalies = detect_anomalies(df)

            # 3. News Sentiment
            sentiment_summary = get_aggregate_sentiment(symbol)
            sentiment_score = sentiment_summary.get("score", 0.0)

            # 4. Smart Money
            sm_report = self.sm_analyst.analyze_floorsheet(None, recent_ohlcv=df)

            # 5. Market Regime
            regime_info = self.regime_detector.detect_regime(df, sm_report)

            # 6. ML Training & Forecast
            feat_df = build_features(df, sentiment_score=sentiment_score, smart_money_info=sm_report)
            feat_df = add_targets(feat_df)
            feature_cols = get_feature_cols(feat_df)
            
            self.ensemble.symbol = symbol 
            self.ensemble.fit(feat_df, feature_cols)
            
            ml_preds = self.ensemble.forecast(feat_df, feature_cols, horizon=n_predict, sentiment_score=sentiment_score)

            # 7. Final Signal & Strategy
            atr_value = trend_info.get("atr")
            trade_plan = self.strat_engine.generate_strategy(
                last_close, 
                ml_preds, 
                sm_report, 
                regime_info, 
                atr=atr_value
            )

            # 8. Unified Decision Score
            m_prob = float(ml_preds[0].direction_prob)
            t_score = float(trend_info.get("score", 0.5))
            s_score = float(sentiment_score)
            mom_score = float(trend_info.get("recent_5d_momentum", 0))
            vol_score = float(trend_info.get("vol_ratio", 0.5))
            
            try:
                final_sig, final_conf = self.ensemble.compute_final_signal(
                    m_prob, t_score, s_score, mom_score, vol_score
                )
            except:
                # Fallback if method not available
                score = 0.35*m_prob + 0.25*t_score + 0.15*(s_score+1)/2 + 0.15*(mom_score+1)/2 + 0.10*vol_score
                final_sig = "BUY" if score > 0.6 else "SELL" if score < 0.4 else "HOLD"
                final_conf = float(score)

            # 9. Scenario Analysis
            scenarios = []
            try:
                if hasattr(self.strat_engine, 'generate_scenarios'):
                    scenarios = self.strat_engine.generate_scenarios(last_close, ml_preds)
                else:
                    # Build default scenarios
                    base_target = float(ml_preds[-1].predicted_close) if ml_preds else last_close
                    scenarios = [
                        {"label": "Bull Case", "probability": 30, "target": round(last_close * 1.08, 2), "change_pct": 8.0},
                        {"label": "Base Case", "probability": 50, "target": round(base_target, 2), "change_pct": round((base_target/last_close - 1)*100, 2)},
                        {"label": "Bear Case", "probability": 20, "target": round(last_close * 0.90, 2), "change_pct": -10.0},
                    ]
            except Exception as e:
                logger.warning(f"Scenario generation failed: {e}")

            # 10. Model reliability reason
            model_reliability = 0.0
            if self.ensemble.report_:
                dir_acc = getattr(self.ensemble.report_, 'avg_dir_acc', 0)
                model_reliability = round(dir_acc, 1)
            
            signal_reason = (
                f"ML Probability: {int(m_prob*100)}%, "
                f"Trend: {trend_info.get('trend_label', 'NEUTRAL')}, "
                f"Sentiment: {'POS' if sentiment_score > 0 else 'NEG'}"
            )

            # 11. AI Analyst Summary
            regime_name = regime_info.get("regime", "NEUTRAL")
            ml_target = float(ml_preds[-1].predicted_close) if ml_preds else 0
            ai_summary = (
                f"{symbol} shows a {regime_name} market regime. "
                f"The recommended stance is {final_sig} with {final_conf*100:.1f}% confidence. "
                f"RSI at {trend_info.get('rsi', '--')} indicates {trend_info.get('rsi_label', 'neutral')}. "
                f"ML ensemble targets NPR {ml_target:,.2f} over {n_predict} sessions. "
                f"Volatility is {trend_info.get('volatility_label', 'NORMAL')} ({trend_info.get('volatility_pct', 0)}%). "
                "Calculated from ensemble of 5 models (LGB, XGB, GBM, RF, Ridge)."
            )

            # 12. Accuracy Metrics
            accuracy_dict = {}
            if self.ensemble.report_:
                report = self.ensemble.report_
                accuracy_dict = {
                    "avg_mae": round(getattr(report, 'avg_mae', 0), 2),
                    "avg_rmse": round(getattr(report, 'avg_rmse', 0), 2),
                    "avg_dir_acc": round(getattr(report, 'avg_dir_acc', 0), 1),
                    "avg_mape": round(getattr(report, 'avg_mape', 0), 2),
                    "folds": [
                        {
                            "fold": f.fold,
                            "mae": round(f.mae, 2),
                            "rmse": round(f.rmse, 2),
                            "dir_acc": round(f.dir_acc, 1),
                            "mape": round(f.mape, 2)
                        } for f in getattr(report, 'cv_folds', [])
                    ]
                }

            # 13. Top Features
            top_features = []
            if self.ensemble.report_ and getattr(self.ensemble.report_, 'feature_importance', None):
                for k, v in self.ensemble.report_.feature_importance.items():
                    top_features.append({"name": k, "value": round(float(v), 4)})

            # 14. Serialize forecast with ALL fields
            forecast_serialized = serialize_forecast(ml_preds)

            return standardize({
                "symbol": symbol.upper(),
                # Dataset summary
                "last_price": last_close,
                "prev_close": prev_close,
                "week52_high": week52_high,
                "week52_low": week52_low,
                "avg_vol_20d": avg_vol_20d,
                "date_range_start": date_range_start,
                "date_range_end": date_range_end,
                "records_count": records_count,
                # Core ML data
                "trend_info": trend_info,
                "anomalies": anomalies,
                "sentiment": sentiment_summary,
                "regime": regime_info,
                "smart_money": sm_report,
                "forecast": forecast_serialized,
                "trade_plan": trade_plan,
                "final_signal": final_sig,
                "final_confidence": final_conf,
                "model_reliability": model_reliability,
                "signal_reason": signal_reason,
                # New enriched data
                "scenarios": scenarios,
                "accuracy_metrics": accuracy_dict,
                "top_features": top_features,
                "ai_summary": ai_summary,
                "is_success": True
            })

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}", exc_info=True)
            return {"symbol": symbol, "is_success": False, "error": str(e)}

    def get_all_stocks(self):
        return fetch_company_list()
