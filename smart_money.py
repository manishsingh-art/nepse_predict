import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SmartMoneyAnalyst:
    """
    Analyzes historical and intraday broker activity to identify 
    'Smart Money' behavior (accumulation, distribution, wash-trading).
    """
    
    def calculate_broker_hhi(self, broker_volumes: pd.Series) -> float:
        """
        Calculates the Herfindahl-Hirschman Index (HHI) for broker concentration.
        HHI < 1500: Unconcentrated (Retail dominated)
        1500 - 2500: Moderate concentration
        > 2500: Highly concentrated (Institutional/Operator dominated)
        """
        if broker_volumes.empty or broker_volumes.sum() == 0:
            return 0.0
        shares = (broker_volumes / broker_volumes.sum()) * 100
        return float((shares**2).sum())

    def analyze_floorsheet(self, floorsheet_df: pd.DataFrame, recent_ohlcv: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Deeper analysis of the floorsheet.
        Detects:
        - Broker Concentration Index (HHI)
        - Absorption & Hidden Accumulation
        - Trap Probability (Fake Breakouts)
        """
        if floorsheet_df is None or floorsheet_df.empty:
            return {"status": "No data", "regime": "RETAIL", "trap_score": 0}
            
        try:
            df = floorsheet_df.copy()
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
            df["rate"] = pd.to_numeric(df["rate"], errors="coerce").fillna(0)
            
            total_vol = df["quantity"].sum()
            total_trades = len(df)
            avg_trade_size = total_vol / total_trades if total_trades > 0 else 0
            
            # 1. Concentration Analysis
            buyers = df.groupby("buyer_broker")["quantity"].sum().sort_values(ascending=False)
            sellers = df.groupby("seller_broker")["quantity"].sum().sort_values(ascending=False)

            # Broker net flow (buy_qty - sell_qty per broker)
            flow = buyers.subtract(sellers, fill_value=0.0).sort_values(ascending=False)
            net_flow_top10 = float(flow.head(10).sum())
            net_flow_top10_norm = float(net_flow_top10 / (total_vol + 1e-9))
            broker_concentration = float(top5_buy_share - top5_sell_share)
            accumulation_flag = float(top5_buy_share > 0.40 and net_flow_top10_norm > 0)
            
            buy_hhi = self.calculate_broker_hhi(buyers)
            sell_hhi = self.calculate_broker_hhi(sellers)
            
            top5_buy_share = buyers.head(5).sum() / total_vol if total_vol > 0 else 0
            top5_sell_share = sellers.head(5).sum() / total_vol if total_vol > 0 else 0
            
            # 2. Broker Divergence (Manipulation Signal)
            wash_signal = False
            if not buyers.empty and not sellers.empty:
                if buyers.index[0] == sellers.index[0]:
                    if (buyers.iloc[0] / total_vol > 0.15) and (sellers.iloc[0] / total_vol > 0.15):
                        wash_signal = True
            
            # 3. Hidden Accumulation Detection
            # Price stable (low volatility) but broker concentration rising
            hidden_accumulation = False
            if recent_ohlcv is not None and len(recent_ohlcv) >= 5:
                price_std = recent_ohlcv["close"].tail(5).std() / recent_ohlcv["close"].iloc[-1]
                if price_std < 0.01 and buy_hhi > 2000: # Tight price, strong buying concentration
                    hidden_accumulation = True

            # 4. Large Trade Participation
            large_trades = df[df["quantity"] > avg_trade_size * 5]
            large_trade_ratio = large_trades["quantity"].sum() / total_vol if total_vol > 0 else 0
            
            # 5. Trap Probability Score
            trap_score = self.calculate_trap_score(floorsheet_df, recent_ohlcv)
            
            regime = "RETAIL"
            if top5_buy_share > self.accumulation_threshold:
                regime = "ACCUMULATION 🏦"
            if top5_sell_share > self.accumulation_threshold:
                regime = "DISTRIBUTION 📉"
            if hidden_accumulation:
                regime = "HIDDEN ACCUMULATION 🕵️‍♂️"
            if wash_signal:
                regime = "POTENTIAL MANIPULATION ⚡"
            if large_trade_ratio > 0.3:
                regime = "OPERATOR ACTIVE 🐋"
            
            return {
                "regime": regime,
                "buy_hhi": round(buy_hhi, 1),
                "sell_hhi": round(sell_hhi, 1),
                "buy_concentration": round(top5_buy_share, 3),
                "sell_concentration": round(top5_sell_share, 3),
                "broker_concentration": round(broker_concentration, 3),
                "smart_money_flow": round(net_flow_top10_norm, 4),
                "accumulation_flag": accumulation_flag,
                "large_trade_participation": round(large_trade_ratio, 3),
                "wash_trading_alert": wash_signal,
                "hidden_accumulation": hidden_accumulation,
                "trap_score": trap_score,
                "avg_trade_size": round(avg_trade_size, 1),
                "top_buying_broker": int(buyers.index[0]) if not buyers.empty else None,
                "top_selling_broker": int(sellers.index[0]) if not sellers.empty else None
            }
        except Exception as e:
            logger.error(f"SmartMoney analysis error: {e}")
            return {"status": "Error", "message": str(e), "regime": "ERROR", "trap_score": 0}

    def calculate_trap_score(self, df: pd.DataFrame, ohlcv: Optional[pd.DataFrame]) -> int:
        """
        Calculates a 'Trap Probability' score (0-100).
        High score means the current move is likely a fake breakout/pump.
        Parameters:
        - Price jumping without large-trade participation.
        - High RSI + Thin volume.
        - Same broker leading both buy and sell.
        """
        score = 0
        if df is None or df.empty: return 0
        
        try:
            total_vol = df["quantity"].sum()
            # 1. Large trade validation (Real moves have institutional backing)
            avg_size = total_vol / len(df)
            large_vol = df[df["quantity"] > avg_size * 5]["quantity"].sum()
            if large_vol / total_vol < 0.1: score += 30 # No big hands
            
            if ohlcv is not None and not ohlcv.empty:
                last_row = ohlcv.iloc[-1]
                # 2. RSI Overextended
                if last_row.get("rsi_14", 50) > 75: score += 20
                
                # 3. Volume Divergence (Price Up, Vol Down)
                if len(ohlcv) > 1:
                    price_up = last_row["close"] > ohlcv["close"].iloc[-2]
                    vol_down = last_row["volume"] < ohlcv["volume"].rolling(5).mean().iloc[-1]
                    if price_up and vol_down: score += 30
            
            return min(score, 100)
        except:
            return 0

    def calculate_accumulation_distribution_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Modified Chaikin A/D that emphasizes Volume-Price confirmation.
        """
        cl = df["close"]
        hi = df["high"]
        lo = df["low"]
        vol = df["volume"]
        
        mfm = ((cl - lo) - (hi - cl)) / (hi - lo + 1e-9)
        mfv = mfm * vol
        return mfv.cumsum()
