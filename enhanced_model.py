#!/usr/bin/env python3
"""
enhanced_model.py — Lightweight feature plugin layer for NEPSE Predict

This module is intentionally model-agnostic:
- It provides a registry of "behavioral" / "context" features.
- It can compute a stability-oriented confidence score.
- It exposes a simple decision engine to prevent conflicting signals.

It is designed to compose with the existing pipeline (features.py, smart_money.py,
nepse_live.py) rather than replace it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

import math


FeatureFn = Callable[[Mapping[str, Any]], Any]


@dataclass
class EnhancedModel:
    version: str = "2.0"
    features: Dict[str, FeatureFn] = field(default_factory=dict)

    def add_feature(self, name: str, function: FeatureFn) -> None:
        """Register a computed feature function."""
        self.features[name] = function

    def compute_features(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Compute all registered features using `context`.

        `context` can be any mapping (dict-like) carrying precomputed signals such as:
        - OHLCV-derived stats (ranges, zscores)
        - floorsheet-derived smart money stats
        - index/sector sentiment stats
        """
        out: Dict[str, Any] = {}
        for name, fn in self.features.items():
            try:
                out[name] = fn(context)
            except Exception:
                out[name] = None
        return out

    # ── Confidence (stability-oriented) ──────────────────────────────────────
    def calculate_confidence(
        self,
        model_agreement: float,
        volatility: float,
        regime_stability: float,
        prediction_variance: float,
    ) -> float:
        """
        Returns a [0..1] confidence score.

        Inputs are expected to be roughly normalized:
        - model_agreement: 0..1 (e.g., fraction of base models pointing same direction)
        - volatility: 0..1 (higher = more volatile, should reduce confidence)
        - regime_stability: 0..1 (higher = more stable regime)
        - prediction_variance: >=0 (higher = less stable predictions)
        """
        ma = float(max(0.0, min(1.0, model_agreement)))
        rs = float(max(0.0, min(1.0, regime_stability)))
        vol = float(max(0.0, min(1.0, volatility)))
        pv = float(max(0.0, prediction_variance))

        # Penalize volatility and variance; reward agreement and stable regime.
        raw = (0.55 * ma + 0.45 * rs) / (1.0 + 1.75 * pv)
        raw *= (1.0 - 0.60 * vol)
        return float(max(0.0, min(1.0, raw)))

    # ── Leading indicators ───────────────────────────────────────────────────
    def add_leading_indicators(self) -> None:
        """
        Registers leading indicators that aim to precede price movement.

        Expected context keys:
        - current_range, historical_range
        - volume, average_volume
        - sweep_volume, total_volume
        """
        self.add_feature("range_compression_breakout", self.detect_range_compression)
        self.add_feature("volume_before_move", self.detect_volume_before_price)
        self.add_feature("liquidity_sweep", self.detect_liquidity_sweep)

    @staticmethod
    def detect_range_compression(ctx: Mapping[str, Any]) -> bool:
        cur = _to_float(ctx.get("current_range"))
        hist = _to_float(ctx.get("historical_range"))
        if cur is None or hist is None or hist <= 0:
            return False
        return cur < hist * 0.65

    @staticmethod
    def detect_volume_before_price(ctx: Mapping[str, Any]) -> bool:
        vol = _to_float(ctx.get("volume"))
        avg = _to_float(ctx.get("average_volume"))
        if vol is None or avg is None or avg <= 0:
            return False
        return vol > avg * 1.5

    @staticmethod
    def detect_liquidity_sweep(ctx: Mapping[str, Any]) -> bool:
        sweep = _to_float(ctx.get("sweep_volume"))
        total = _to_float(ctx.get("total_volume"))
        if sweep is None or total is None or total <= 0:
            return False
        return sweep > total * 0.25

    # ── Smart money features ────────────────────────────────────────────────
    def enhance_smart_money_analysis(self) -> None:
        """
        Registers floorsheet-derived smart-money features.

        Expected context keys:
        - top_buyers_volume, total_volume
        - broker_top_volumes (list[float]), broker_total_volume
        - accumulation, distribution
        """
        self.add_feature("smart_money_score", self.calculate_smart_money_score)
        self.add_feature("broker_concentration", self.calculate_broker_concentration)
        self.add_feature("net_accumulation", self.calculate_net_accumulation)

    @staticmethod
    def calculate_smart_money_score(ctx: Mapping[str, Any]) -> Optional[float]:
        top = _to_float(ctx.get("top_buyers_volume"))
        total = _to_float(ctx.get("total_volume"))
        if top is None or total is None or total <= 0:
            return None
        return float(top / total)

    @staticmethod
    def calculate_broker_concentration(ctx: Mapping[str, Any]) -> Optional[float]:
        vols = ctx.get("broker_top_volumes")
        total = _to_float(ctx.get("broker_total_volume"))
        if vols is None or total is None or total <= 0:
            return None
        try:
            top_sum = float(sum(float(v) for v in list(vols)))
        except Exception:
            return None
        return float(top_sum / total)

    @staticmethod
    def calculate_net_accumulation(ctx: Mapping[str, Any]) -> Optional[float]:
        acc = ctx.get("accumulation")
        dist = ctx.get("distribution")
        try:
            acc_v = float(acc)
            dist_v = float(dist)
        except Exception:
            return None
        return float(acc_v - dist_v)

    # ── Market index dependency ──────────────────────────────────────────────
    def add_market_index_dependency(self) -> None:
        """
        Registers index dependency features.

        Expected context keys:
        - stock_ret_window (array-like), index_ret_window (array-like)
        - stock_price_change, index_price_change
        - stock_trend, index_trend
        """
        self.add_feature("beta_to_nepse", self.calculate_beta_to_nepse)
        self.add_feature("relative_strength_vs_index", self.calculate_relative_strength)
        self.add_feature("index_trend_alignment", self.calculate_index_trend_alignment)

    @staticmethod
    def calculate_beta_to_nepse(ctx: Mapping[str, Any]) -> Optional[float]:
        sr = ctx.get("stock_ret_window")
        ir = ctx.get("index_ret_window")
        try:
            s = [float(x) for x in list(sr)]
            i = [float(x) for x in list(ir)]
        except Exception:
            return None
        if len(s) < 5 or len(i) < 5 or len(s) != len(i):
            return None
        var = _variance(i)
        if var <= 0:
            return None
        cov = _covariance(s, i)
        return float(cov / var)

    @staticmethod
    def calculate_relative_strength(ctx: Mapping[str, Any]) -> Optional[float]:
        sp = _to_float(ctx.get("stock_price_change"))
        ip = _to_float(ctx.get("index_price_change"))
        if sp is None or ip is None:
            return None
        if abs(ip) < 1e-12:
            return None
        return float(sp / ip)

    @staticmethod
    def calculate_index_trend_alignment(ctx: Mapping[str, Any]) -> Optional[bool]:
        st = ctx.get("stock_trend")
        it = ctx.get("index_trend")
        if st is None or it is None:
            return None
        return bool(st == it)

    # ── News sentiment refinement ────────────────────────────────────────────
    def refine_news_sentiment(self) -> None:
        """
        Registers sentiment-related contextual features.

        Expected context keys:
        - sector_sentiment (scalar), sector_weight
        - news_impact_score (scalar), volume_spike, avg_volume
        """
        self.add_feature("sector_sentiment", self.calculate_sector_sentiment)
        self.add_feature("news_weighted_by_volume", self.calculate_news_impact_by_volume)

    @staticmethod
    def calculate_sector_sentiment(ctx: Mapping[str, Any]) -> Optional[float]:
        s = _to_float(ctx.get("sector_sentiment"))
        w = _to_float(ctx.get("sector_weight"), default=0.5)
        if s is None:
            return None
        w = float(max(0.0, min(1.0, w)))
        return float(s * w)

    @staticmethod
    def calculate_news_impact_by_volume(ctx: Mapping[str, Any]) -> Optional[float]:
        impact = _to_float(ctx.get("news_impact_score"))
        spike = _to_float(ctx.get("volume_spike"))
        avg = _to_float(ctx.get("avg_volume"))
        if impact is None or spike is None or avg is None or avg <= 0:
            return None
        return float(impact * (spike / avg))

    # ── Final decision engine ────────────────────────────────────────────────
    @staticmethod
    def decision_engine(
        model_bearish: bool,
        rsi_overbought: bool,
        low_momentum: bool,
        strong_trend: bool,
        high_volume: bool,
    ) -> str:
        """
        Decide final action based on multiple signals to avoid conflicts.
        """
        if model_bearish and rsi_overbought and low_momentum:
            return "SELL"
        if strong_trend and high_volume:
            return "BUY"
        return "HOLD"


def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return default
    if not math.isfinite(x):
        return default
    return x


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def _covariance(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = _mean(xs)
    my = _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (len(xs) - 1)

