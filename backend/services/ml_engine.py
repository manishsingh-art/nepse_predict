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

from fetcher import fetch_history, fetch_company_list, fetch_live_price, get_aggregate_sentiment
from features import build_features, add_targets, get_feature_cols
from models import NEPSEEnsemble
from smart_money import SmartMoneyAnalyst
from strategy import TradingStrategyEngine
from analyze import add_indicators, detect_trend, detect_anomalies
from regime import MarketRegimeDetector

# Silence the console logs as requested
logging.getLogger("fetcher").setLevel(logging.ERROR)
logging.getLogger("analyze").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

def standardize(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return standardize(obj.tolist())
    if isinstance(obj, dict):
        return {k: standardize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [standardize(i) for i in obj]
    return obj

def run_prediction(symbol: str) -> Dict[str, Any]:
    """
    Runs the full ML pipeline and returns strictly formatted JSON.
    For now, returns mock data for testing.
    """
    try:
        # Mock data for testing
        import random
        from datetime import datetime, timedelta
        
        last_close = 500 + random.randint(-50, 50)  # Mock current price
        
        # Mock forecast
        forecast_list = []
        current_date = datetime.now()
        for i in range(7):
            forecast_list.append({
                "date": (current_date + timedelta(days=i+1)).isoformat(),
                "price": last_close + random.randint(-20, 20),
                "change_pct": random.uniform(-5, 5),
                "prob_up": 50 + random.randint(-20, 20),
                "confidence": random.uniform(0.1, 1.0)
            })
        
        # Mock scenarios
        scenarios = {
            "bull": {"prob": 30, "target": last_close * 1.08},
            "base": {"prob": 50, "target": last_close * 1.02},
            "bear": {"prob": 20, "target": last_close * 0.92}
        }
        
        # Mock trade plan
        trade_plan = {
            "buy_zone": [last_close * 0.98, last_close * 1.02],
            "target": last_close * 1.1,
            "stop_loss": last_close * 0.9,
            "rr_ratio": 2.0
        }
        
        # Mock signal
        signals = ["BUY", "SELL", "HOLD"]
        final_sig = random.choice(signals)
        
        return {
            "symbol": symbol.upper(),
            "predicted_close": last_close + random.randint(-10, 10),
            "signal": final_sig,
            "direction_prob": random.uniform(0.3, 0.7),
            "confidence": random.uniform(0.5, 0.9),
            "forecast": forecast_list,
            "scenarios": scenarios,
            "trade_plan": trade_plan,
            "risks": ["Mock risk 1", "Mock risk 2"],
            "anomalies": [{"date": "2024-01-01", "type": "spike", "change_pct": 5.0}],
            "ai_summary": f"Mock summary for {symbol}",
            "technical": {},
            "sentiment": {},
            "accuracy": {},
            "features": {},
            "is_success": True
        }

    except Exception as e:
        logger.error(f"Error in run_prediction for {symbol}: {str(e)}")
        return {"symbol": symbol, "is_success": False, "error": str(e)}
    except Exception as e:
        logger.error(f"Error in run_prediction for {symbol}: {str(e)}")
        raise

class PredictorService:
    def __init__(self):
        self.sm_analyst = SmartMoneyAnalyst()
        self.strat_engine = TradingStrategyEngine()
        self.ensemble = NEPSEEnsemble(optimise=False) # fast execution
        self.regime_detector = MarketRegimeDetector()

    def run_prediction(self, symbol: str, years: int = 2, n_predict: int = 7) -> Dict[str, Any]:
        """
        Runs the full ML pipeline and returns strictly formatted JSON.
        No print statements.
        """
        try:
            # 1. Fetch Data
            df = fetch_history(symbol, years=years)
            if df.empty:
                raise ValueError(f"No history found for {symbol}")

            live_price = fetch_live_price(symbol)
            if live_price and live_price > 0:
                df.loc[df.index[-1], "close"] = live_price

            last_close = float(df["close"].iloc[-1])
            
            # 2. Technical Analysis
            df = add_indicators(df)
            trend_info = detect_trend(df)
            raw_anomalies = detect_anomalies(df)

            # 3. Sentiment
            sentiment_summary = get_aggregate_sentiment(symbol)
            sentiment_score = float(sentiment_summary.get("score", 0.0))

            # 4. Smart Money & Regime
            sm_report = self.sm_analyst.analyze_floorsheet(None, recent_ohlcv=df)
            regime_info = self.regime_detector.detect_regime(df, sm_report)

            # 5. ML Training & Forecast
            feat_df = build_features(df, sentiment_score=sentiment_score, smart_money_info=sm_report)
            feat_df = add_targets(feat_df)
            feature_cols = get_feature_cols(feat_df)
            
            self.ensemble.symbol = symbol 
            self.ensemble.fit(feat_df, feature_cols)
            ml_preds = self.ensemble.forecast(feat_df, feature_cols, horizon=n_predict, sentiment_score=sentiment_score)

            # 6. Formatting FORECAST
            forecast_list = []
            for pred in ml_preds:
                forecast_list.append({
                    "date": standardize(getattr(pred, "date", datetime.now().date())),
                    "price": float(getattr(pred, "predicted_close", last_close)),
                    "change_pct": float(getattr(pred, "change_pct", 0.0)),
                    "prob_up": float(getattr(pred, "direction_prob", 50.0) * 100), # prob_up in 0-100 range as requested
                    "confidence": float(getattr(pred, "direction_confidence", 0.5) * 10) # 0-10 range as requested
                })

            # 7. Final Signal & Trade Plan
            atr_value = trend_info.get("atr")
            raw_trade_plan = self.strat_engine.generate_strategy(last_close, ml_preds, sm_report, regime_info, atr=atr_value)

            trade_plan = {
                "buy_zone": [
                    float(raw_trade_plan.get("buy_zone_low", last_close * 0.98)),
                    float(raw_trade_plan.get("buy_zone_high", last_close * 1.02))
                ],
                "target": float(raw_trade_plan.get("take_profit", last_close * 1.1)),
                "stop_loss": float(raw_trade_plan.get("stop_loss", last_close * 0.9)),
                "rr_ratio": float(raw_trade_plan.get("risk_reward_ratio", 1.5))
            }

            # 8. Scenarios Formatted strictly
            m_prob = float(ml_preds[0].direction_prob)
            t_score = float(trend_info.get("score", 0.5))
            mom_score = float(trend_info.get("recent_5d_momentum", 0))
            vol_score = float(trend_info.get("vol_ratio", 0.5))
            
            try:
                final_sig, final_conf = self.ensemble.compute_final_signal(m_prob, t_score, sentiment_score, mom_score, vol_score)
            except:
                score = 0.35*m_prob + 0.25*t_score + 0.15*(sentiment_score+1)/2 + 0.15*(mom_score+1)/2 + 0.10*vol_score
                final_sig = "BUY" if score > 0.6 else "SELL" if score < 0.4 else "HOLD"
                final_conf = float(score)

            base_target = float(ml_preds[-1].predicted_close) if ml_preds else last_close
            scenarios = {
                "bull": {"prob": 30, "target": round(last_close * 1.08, 2)},
                "base": {"prob": 50, "target": round(base_target, 2)},
                "bear": {"prob": 20, "target": round(last_close * 0.90, 2)}
            }

            # 9. Format Anomalies
            formatted_anomalies = []
            for a in raw_anomalies:
                formatted_anomalies.append({
                    "date": standardize(a.get("date", datetime.now().date())),
                    "type": a.get("label", "spike"),
                    "change_pct": float(a.get("change_pct", 0.0))
                })

            # 10. AI Summary and Risks
            rsi_val = float(trend_info.get("rsi", 50))
            ai_summary = f"{symbol} currently indicates a {final_sig} signal with {final_conf*100:.1f}% confidence. The trend is {trend_info.get('trend_label', 'neutral')}."
            
            risks = []
            if rsi_val > 70: risks.append("RSI overbought")
            if sentiment_score < 0: risks.append("Negative sentiment")
            if bool(trend_info.get("macd_bullish")) == False: risks.append("MACD is bearish")

            return standardize({
                "symbol": symbol.upper(),
                "predicted_close": float(ml_preds[0].predicted_close) if ml_preds else last_close,
                "signal": final_sig,
                "direction_prob": float(m_prob),
                "confidence": float(final_conf),
                "forecast": forecast_list,
                "scenarios": scenarios,
                "trade_plan": trade_plan,
                "risks": risks,
                "anomalies": formatted_anomalies,
                "ai_summary": ai_summary,
                
                # Additional data for storage
                "technical": trend_info,
                "sentiment": sentiment_summary,
                "accuracy": {"final_conf": final_conf},
                "features": {"feature_cols": feature_cols},
                
                # Hidden extra properties for the API state return if necessary
                "is_success": True
            })

        except Exception as e:
            logger.error(f"Prediction error for {symbol}: {e}", exc_info=False)
            return {"symbol": symbol, "is_success": False, "error": str(e)}

    def get_all_stocks(self):
        return fetch_company_list()
