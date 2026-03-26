import logging
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class TradingStrategyEngine:
    """
    v6.1 Trading Strategy Engine.
    - ATR-based dynamic Stop-Loss
    - Volatility-adjusted position sizing
    - Manipulation trap filter
    - Regime-context aware decisions
    """

    def __init__(self, risk_per_trade: float = 0.02):
        self.risk_per_trade = risk_per_trade  # 2% account risk per trade

    def generate_strategy(
        self,
        current_price: float,
        forecasts: List[Any],
        smart_money: Dict[str, Any],
        regime: Dict[str, Any],
        atr: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        v6.1 strategy generation.
        - Uses ATR for dynamic SL/TP if available
        - Volatility-adjusted position sizing
        """
        if not forecasts:
            return {"action": "HOLD", "reason": "No forecast data", "stop_loss": 0, "take_profit": 0, "risk_reward_ratio": 0, "suggested_size_weight": 0}

        next_day = forecasts[0]
        prob = next_day.direction_prob
        conf = getattr(next_day, "direction_confidence", 0.5)
        trap = getattr(next_day, "trap_score", 0)
        regime_name = regime.get("regime", "UNKNOWN")
        vol_pct = regime.get("volatility_pct", 2.0) / 100.0

        # ── 1. Manipulation Filter ─────────────────────────────────────────────
        if trap > 45:
            return {
                "action": "AVOID / EXIT",
                "reason": f"High Manipulation Risk (Trap Score: {trap})",
                "risk_level": "CRITICAL",
                "regime_context": regime_name,
                "stop_loss": round(current_price * 0.97, 2),
                "take_profit": round(current_price, 2),
                "risk_reward_ratio": 0.0,
                "suggested_size_weight": 0.0,
                "trap_index": trap,
            }

        # ── 2. Decision Logic ──────────────────────────────────────────────────
        action = "HOLD / NEUTRAL"
        if prob > 0.55 and conf > 0.15:
            action = "BUY" if ("BULL" in regime_name or "ACCUMULATION" in regime_name) else "SPECULATIVE BUY"
        elif prob < 0.45 and conf > 0.15:
            action = "SELL / REDUCE"

        # ── 3. ATR-based SL/TP (v6.1) ─────────────────────────────────────────
        # ATR gives a realistic daily range for the stock
        if atr and atr > 0:
            sl_distance = atr * 1.5           # SL = 1.5x ATR below entry
            tp_distance = atr * 2.5           # TP = 2.5x ATR above entry (RR ~1.67)
        else:
            # Fallback to volatility band from forecast
            sl_distance = current_price * vol_pct * 1.5
            tp_distance = current_price * vol_pct * 2.5

        if "SELL" in action:
            sl = round(current_price + sl_distance, 2)  # SL above for shorts
            tp = round(current_price - tp_distance, 2)
        else:
            sl = round(current_price - sl_distance, 2)
            tp = round(current_price + tp_distance, 2)

        # Extend TP to 3rd-day forecast if bullish and it's higher
        if len(forecasts) >= 3 and "BUY" in action:
            f3 = forecasts[2]
            if f3.predicted_close > tp:
                tp = round(f3.predicted_close, 2)

        rr_ratio = (tp - current_price) / (sl_distance + 1e-9)

        # ── 4. Volatility-Adjusted Position Sizing (v6.1) ─────────────────────
        # Higher volatility → smaller position; higher confidence → larger position
        base_size = self.risk_per_trade / (vol_pct + 1e-9)
        size_multiplier = float(np.clip(base_size * conf * 2, 0.25, 2.0))

        if "AVOID" in action or "HOLD" in action:
            size_multiplier = 0.0

        return {
            "action": action,
            "entry": round(current_price, 2),
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward_ratio": round(rr_ratio, 2),
            "suggested_size_weight": round(size_multiplier, 2),
            "regime_context": regime_name,
            "sm_context": smart_money.get("regime", "Retail"),
            "trap_index": trap,
            "atr_used": round(atr, 2) if atr else None,
            "reason": f"P(up)={round(prob,2)} | Conf={round(conf,2)} | Regime: {regime_name}",
        }
