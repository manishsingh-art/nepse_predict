import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Classifies the current market state into major regimes and transitional states (v6.2):
    1. BULL (Trending up, good volume, low volatility)
    2. BEAR (Trending down, high volume, high volatility)
    3. SIDEWAYS (No trend, low volume, low volatility)
    4. MANIPULATION/PUMP (Extreme volume spikes, broker concentration)
    
    Now includes Volume Profiles (5d vs 20d) and EMA Velocity for early transition detection.
    """
    
    def __init__(self, lookback: int = 20):
        self.lookback = lookback
        
    def detect_regime(self, df: pd.DataFrame, smart_money_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Input: DataFrame with OHLCV data.
        Returns: {regime, confidence, color}
        """
        if len(df) < self.lookback:
            return {"regime": "NEUTRAL", "confidence": 0, "color": "white"}
            
        try:
            cl = df["close"].astype(float)
            vol = df["volume"].astype(float)
            
            # 1. Trend (EMA alignment & Velocity for transitions)
            ema10 = cl.ewm(span=10, adjust=False).mean()
            ema20 = cl.ewm(span=20, adjust=False).mean()
            ema50 = cl.ewm(span=50, adjust=False).mean()
            
            last_close = cl.iloc[-1]
            last_ema10 = ema10.iloc[-1]
            last_ema20 = ema20.iloc[-1]
            last_ema50 = ema50.iloc[-1]
            
            # EMA Velocity (Slope)
            ema20_slope = (last_ema20 - ema20.iloc[-5]) / ema20.iloc[-5]
            
            is_uptrend = last_close > last_ema20 > last_ema50
            is_downtrend = last_close < last_ema20 < last_ema50
            
            # 2. Volatility (ATR relative to price)
            hi, lo = df["high"].astype(float), df["low"].astype(float)
            tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (lo - cl.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(self.lookback).mean().iloc[-1]
            volatility_pct = (atr / last_close) * 100
            
            # 3. Multi-Timeframe Volume Profile (v6.2)
            vol_sma_short = vol.rolling(5).mean().iloc[-1]
            vol_sma_long = vol.rolling(self.lookback).mean().iloc[-1]
            vol_ratio = vol.iloc[-1] / (vol_sma_long + 1e-9)
            vol_momentum = vol_sma_short / (vol_sma_long + 1e-9) # Rising volume profile
            
            # 4. Smart Money Nuance
            trap_score = smart_money_info.get("trap_score", 0) if smart_money_info else 0
            buy_hhi = smart_money_info.get("buy_hhi", 0) if smart_money_info else 0
            
            # 5. Final Classification (Transition Aware)
            regime = "SIDEWAYS"
            confidence = 0.5
            color = "yellow"
            
            if is_uptrend:
                if trap_score > 50:
                    regime = "BULL (Bull Trap / Exhaustion)"
                    color = "cyan"
                    confidence = 0.4
                elif ema20_slope < 0.001 and vol_momentum < 0.8:
                    regime = "BULL ➔ Transitioning to SIDEWAYS"
                    color = "magenta"
                    confidence = 0.6
                elif last_close < last_ema10 and vol_ratio > 1.2:
                    regime = "BULL ➔ Transitioning to BEAR (Distribution)"
                    color = "red"
                    confidence = 0.5
                elif vol_ratio > 1.2 and volatility_pct < 3.5:
                    regime = "BULL (Healthy Accumulation)"
                    color = "green"
                    confidence = 0.8
                elif volatility_pct > 5.5:
                    regime = "BULL (Parabolic / Overextended)"
                    color = "bright_green"
                    confidence = 0.6
                else:
                    regime = "BULL (Gradual)"
                    color = "green"
                    confidence = 0.7
            elif is_downtrend:
                if vol_ratio > 1.8:
                    regime = "BEAR (Panic / Capitulation)"
                    color = "red"
                    confidence = 0.9
                elif last_close > last_ema10 and vol_momentum > 1.2:
                    regime = "BEAR ➔ Transitioning to BULL (Bottoming)"
                    color = "bright_green"
                    confidence = 0.6
                elif volatility_pct < 2.5:
                    regime = "BEAR (Slow Bleed / Distribution)"
                    color = "magenta"
                    confidence = 0.7
                else:
                    regime = "BEAR (Trending Down)"
                    color = "red"
                    confidence = 0.8
            elif trap_score > 60 or (vol_ratio > 3 and smart_money_info and smart_money_info.get("wash_trading_alert")):
                regime = "MANIPULATION / PUMP-AND-DUMP"
                color = "bright_red"
                confidence = 0.5
            elif volatility_pct < 1.5 and vol_ratio < 0.8:
                if buy_hhi > 2000:
                    regime = "ACCUMULATION (Hidden / Quiet)"
                    color = "blue"
                    confidence = 0.7
                elif last_close > last_ema10 and ema20_slope > 0.005:
                    regime = "SIDEWAYS ➔ Transitioning to BULL (Breakout Watch)"
                    color = "cyan"
                    confidence = 0.6
                else:
                    regime = "SIDEWAYS (Dead / Consolidation)"
                    color = "white"
                    confidence = 0.6
                
            return {
                "regime": regime,
                "confidence": round(confidence, 2),
                "color": color,
                "volatility_pct": round(volatility_pct, 2),
                "vol_ratio": round(vol_ratio, 2),
                "trap_score": trap_score
            }
        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return {"regime": "ERROR", "confidence": 0, "color": "white", "trap_score": 0}
