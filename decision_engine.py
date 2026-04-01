from __future__ import annotations

"""
decision_engine.py
------------------
Unifies model forecast + technical signals + regime + sentiment into one action.
Ensures a single final signal is printed and logged.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


Action = str  # "BUY" | "SELL" | "HOLD" | "AVOID"


@dataclass(frozen=True)
class DecisionInputs:
    direction_prob: float                 # P(up) from model (0..1)
    expected_ret_5d_pct: float            # expected % move over ~5 sessions
    regime: str
    regime_confidence: float              # 0..1
    sentiment_score: float                # -1..1
    trap_score: float                     # 0..100
    volatility_pct: float                 # ATR% or regime vol%
    illiquid_flag: float = 0.0            # 0/1
    technical_signals: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class FinalDecision:
    action: Action
    confidence: float                     # 0..1
    score: float                          # signed composite score
    rationale: str
    components: Dict[str, float]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), lo, hi))


def _tech_score(signals: Optional[List[Dict[str, Any]]]) -> float:
    """
    Convert suggest_strategy signal list into a signed score in [-1..+1].
    """
    if not signals:
        return 0.0
    score = 0.0
    w_map = {"STRONG": 0.35, "MODERATE": 0.20, "TECHNICAL": 0.15, "MODEL": 0.20, "CAUTION": 0.10, "NEUTRAL": 0.05}
    for s in signals:
        sig = str(s.get("signal", "")).upper()
        strength = str(s.get("strength", "")).upper()
        w = float(w_map.get(strength, 0.10))
        if "BUY" in sig:
            score += w
        elif "SELL" in sig or "AVOID" in sig or "REDUCE" in sig:
            score -= w
    return _clip(score, -1.0, 1.0)


def compute_final_decision(inp: DecisionInputs) -> FinalDecision:
    # Model score: centered at 0.5 with mild amplification.
    model = _clip((float(inp.direction_prob) - 0.5) * 2.0, -1.0, 1.0)

    # Expected return score: saturate at +/-8% over 5 days.
    ret = _clip(float(inp.expected_ret_5d_pct) / 8.0, -1.0, 1.0)

    tech = _tech_score(inp.technical_signals)

    # Regime score: bullish regimes bias positive, bear regimes negative.
    r = str(inp.regime).upper()
    if "BEAR" in r:
        regime = -0.6
    elif "BULL" in r or "ACCUMULATION" in r:
        regime = 0.6
    elif "MANIPULATION" in r or "PUMP" in r:
        regime = -0.2
    else:
        regime = 0.0
    regime *= _clip(float(inp.regime_confidence), 0.0, 1.0)
    regime = _clip(regime, -1.0, 1.0)

    sentiment = _clip(float(inp.sentiment_score), -1.0, 1.0)

    # Penalties: trap, volatility, illiquidity -> reduce aggressiveness.
    trap_pen = _clip(float(inp.trap_score) / 100.0, 0.0, 1.0)
    vol_pen = _clip(float(inp.volatility_pct) / 8.0, 0.0, 1.0)  # 8% ATR is very high
    illiq_pen = _clip(float(inp.illiquid_flag), 0.0, 1.0)

    # Composite (weights tuned for low-liquidity markets)
    components = {
        "model": model,
        "ret": ret,
        "tech": tech,
        "regime": regime,
        "sentiment": sentiment,
        "trap_pen": -trap_pen,
        "vol_pen": -vol_pen,
        "illiquid_pen": -illiq_pen,
    }
    score = (
        0.38 * model +
        0.22 * ret +
        0.18 * tech +
        0.14 * regime +
        0.08 * sentiment
        - 0.22 * trap_pen
        - 0.10 * vol_pen
        - 0.08 * illiq_pen
    )

    # Confidence: magnitude adjusted down by penalties
    conf = abs(score)
    conf *= (1.0 - 0.45 * trap_pen) * (1.0 - 0.25 * illiq_pen)
    conf = _clip(conf, 0.0, 1.0)

    # Action thresholds
    if trap_pen > 0.75 and conf < 0.55:
        action = "AVOID"
    elif score >= 0.20:
        action = "BUY"
    elif score <= -0.20:
        action = "SELL"
    else:
        action = "HOLD"

    rationale = (
        f"score={score:+.2f} (model={model:+.2f}, tech={tech:+.2f}, "
        f"regime={regime:+.2f}, sent={sentiment:+.2f}, trap={trap_pen:.2f})"
    )

    return FinalDecision(action=action, confidence=conf, score=float(score), rationale=rationale, components=components)

