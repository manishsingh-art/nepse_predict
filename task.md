# Task Tracker — NEPSE Predictor Upgrades

## v6.1 — Advanced Analytics
- [x] **System Architecture & Resilience**
  - [x] Fix `ModelReport` NameError path: add prediction-time guards in `models.py`
  - [x] Update `implementation_plan.md` for v6.1
  - [x] Update `task.md`
- [x] **Behavioral & Volatility Intelligence**
  - [x] Implement GARCH-style volatility clustering in `features.py`
  - [x] Add Price–Volume Divergence analysis in `features.py`
  - [x] Integrate Smart Money HHI as a model feature
- [x] **NLP Sentiment & Intelligence**
  - [x] Upgrade `ollama_ai.py` sentiment prompt (strict JSON + category)
  - [x] Ensure `analyze_sentiment_headlines` is imported into `nepse_live.py`
- [ ] **Verification & Reporting**
  - [x] Run end-to-end verification with `SNLI` (Exit 0, ML training success)
  - [x] Update walkthrough with v6.1 performance metrics (from SNLI run)

## v6.2 — Advanced Resilience & Anomaly Integration
- [x] **Prompt 1: Model Resilience & Error Handling**
  - [x] Wrap base estimators with try/except in `models.py` (already present)
  - [x] Implement fallback logic for NaN predictions (added)
  - [x] Enhance Optuna timeout handling (already present)
- [x] **Prompt 2 & 3: Advanced Anomaly & Sentiment Integration**
  - [x] Upgrade `detect_anomalies` in `analyze.py` to correlate news data
  - [x] Ensure news category output (Macro/Corporate/Sector) in `ollama_ai.py` when available
  - [x] Surface news-linked anomalies in `nepse_live.py` terminal output
- [x] **Prompt 4: Dynamic Market Regime Adaptation**
  - [x] Detect regime transitions in `regime.py` (already implemented)
  - [x] Multi-timeframe moving average volume profiles (already implemented)
- [x] **Prompt 5: Volatility-Adjusted Risk Framework**
  - [x] Strategy uses ATR for TP/SL distances (already implemented)
- [x] **Prompt 6: Advanced Model Optimization**
  - [x] Optuna timeout fallback (covered in Prompt 1)
  - [x] Feature penalty weights for Nepal calendar features in LGBM (present)
- [ ] **PC Test (no Ollama model available)**
  - [x] Run resilience test with `NIFRA` (ensure `--ollama` gracefully falls back)

