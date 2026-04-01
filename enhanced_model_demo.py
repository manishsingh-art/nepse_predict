#!/usr/bin/env python3

from enhanced_model import EnhancedModel


def main():
    m = EnhancedModel(version="2.1")
    m.add_leading_indicators()
    m.enhance_smart_money_analysis()
    m.add_market_index_dependency()
    m.refine_news_sentiment()

    # Minimal, dict-based context example (adapt these keys to your pipeline outputs)
    ctx = {
        "current_range": 8.0,
        "historical_range": 20.0,
        "volume": 1_900_000,
        "average_volume": 1_000_000,
        "sweep_volume": 320_000,
        "total_volume": 1_000_000,
        "top_buyers_volume": 240_000,
        "broker_top_volumes": [140_000, 60_000, 30_000],
        "broker_total_volume": 1_000_000,
        "accumulation": 12.0,
        "distribution": 7.5,
        "stock_ret_window": [0.01, -0.003, 0.007, 0.002, -0.001, 0.004],
        "index_ret_window": [0.008, -0.002, 0.005, 0.001, -0.0005, 0.003],
        "stock_price_change": 0.032,
        "index_price_change": 0.018,
        "stock_trend": "UP",
        "index_trend": "UP",
        "sector_sentiment": 0.25,
        "sector_weight": 0.5,
        "news_impact_score": 0.6,
        "volume_spike": 1_900_000,
        "avg_volume": 1_000_000,
    }

    feats = m.compute_features(ctx)
    print("Enhanced features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")

    action = m.decision_engine(
        model_bearish=True,
        rsi_overbought=True,
        low_momentum=True,
        strong_trend=False,
        high_volume=False,
    )
    print("\nDecision:", action)


if __name__ == "__main__":
    main()

