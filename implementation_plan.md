# NEPSE Predictor — Implementation Plan

## v6.1 — Advanced Analytics (upgrade checklist)

### System Architecture & Resilience
- **Fix `ModelReport` NameError**: validated `models.py` imports and `ModelReport` usage; added additional runtime guards in `predict_one()` to prevent meta-imputer/NaN failures from cascading.
- **Docs**: this plan file added for v6.1 tracking.

### Behavioral & Volatility Intelligence
- **GARCH-style volatility clustering**: implemented/extended in `features.py` via `garch_vol`, `vol_of_vol`, and `garch_cluster`.
- **Price–Volume Divergence**: implemented in `features.py` via `pv_divergence`, `pv_confirmation`, and `pv_divergence_score`.
- **Smart Money HHI as model feature**: injected floorsheet-derived context into `features.py` as constant columns:
  - `sm_buy_hhi`, `sm_sell_hhi`, `sm_buy_concentration`, `sm_sell_concentration`, `sm_trap_score`, `sm_wash_trading_alert`.

### NLP Sentiment & Intelligence
- **Deep-reasoning sentiment prompts**: upgraded `ollama_ai.py` prompt to produce strict JSON sentiment + category with better instruction constraints.
- **`analyze_sentiment_headlines` integration**: already imported/used in `nepse_live.py` with automatic fallback when Ollama isn’t installed/running.

### Verification & Reporting
- **End-to-end verification target**: `python nepse_live.py --symbol SNLI --fast` should train ML and exit 0 (without requiring Ollama).
- **Walkthrough metrics**: update README/walkthrough after running the verification with your machine’s metrics.

---

## v6.2 — Advanced Resilience & Anomaly Integration (upgrade checklist)

### Prompt 1: Model Resilience & Error Handling
- **Try/except for base estimators**: present in `models.py` training loop; extended prediction-time guards to avoid crashes when meta-learner prerequisites are missing.
- **Fallback for NaN predictions**: implemented in `NEPSEEnsemble.predict_one()` (falls back to blended base prediction, then `last_close_`).
- **Optuna timeout handling**: already wrapped; retains default params on failure.

### Prompt 2 & 3: Advanced Anomaly & Sentiment Integration
- **News-aware anomalies**: `detect_anomalies()` in `analyze.py` now normalizes RSS news (`title`, `published_dt`) and correlates anomalies to the closest prior news within a short window.
- **News category (Macro/Corporate/Sector)**: sentiment JSON schema requires one of these categories in `ollama_ai.py` (when Ollama is available).
- **Surface news-linked anomalies in terminal**: `nepse_live.py` already shows a “News Cause” column when any anomaly has a cause.

### Prompt 4: Dynamic Market Regime Adaptation
- `regime.py` already includes transitional regimes and multi-timeframe volume profile signals.

### Prompt 5: Volatility-Adjusted Risk Framework
- Strategy layer already uses ATR for TP/SL distances (as noted in README).

### Prompt 6: Advanced Model Optimization
- Feature penalty weights for Nepal calendar features are applied via `feature_penalty` in LightGBM (prefers `np_` features).

