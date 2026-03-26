# NEPSE Predictor v6.1 (Quantum Advanced Analytics) 🚀

An elite, regime-adaptive intelligence engine for the Nepal Stock Exchange (NEPSE). Built for low-liquidity, sentiment-driven markets — v6.1 extends v6.0's profit-optimized core with advanced volatility modeling, ATR-based risk management, and NLP-driven sentiment analysis.

## 🌟 Key Features

### v6.0 — Core Intelligence
-   **Smart Money Intelligence**: Real-time floorsheet scraping with **Broker HHI (Concentration Index)**, hidden accumulation detection, and **Manipulation Trap Scores** (0–100).
-   **Regime-Aware Meta-Learning**: Classifies 4 market phases (`BULL`, `BEAR`, `SIDEWAYS`, `MANIPULATION`) with regime-specific model "heads".
-   **Behavioral Feature Engineering**: FOMO Index and Panic Index to capture retail sentiment extremes.
-   **Realism Engine v2**: Exponential Growth Penalty + Support/Resistance gravity for realistic forecasts.
-   **Sharpe-Optimized Ensemble**: Hyperparameter tuning (Optuna) prioritizes Sharpe Ratio over raw MAE.

### v6.1 — Advanced Analytics
-   **GARCH Volatility Clustering** (`garch_vol`, `vol_of_vol`): Detects volatility regime changes before they hit price.
-   **Price-Volume Divergence** (`pv_divergence_score`): Flags exhaustion moves where volume doesn't confirm price.
-   **ATR-Based Risk Management**: Stop-loss = 1.5× ATR, Take-Profit = 2.5× ATR — fully dynamic per market conditions.
-   **Volatility-Adjusted Position Sizing**: Auto-scales trade size based on predicted risk and directional confidence.
-   **NLP Sentiment via Ollama**: `analyze_sentiment_headlines()` uses Llama3 to score news sentiment (-1.0 to +1.0) with structured JSON output.

---

## 🛠 Setup & Installation

### 1. Requirements
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. (Optional) AI / NLP Setup
```bash
# Install Ollama and pull Llama3 for NLP sentiment & AI analyst
ollama pull llama3
```

---

## 🚀 How to Run

### Standard (ML Ensemble)
```bash
python nepse_live.py --symbol NABIL --predict 7
```

### AI Enhanced (With Ollama — NLP Sentiment + Analyst Summary)
```bash
python nepse_live.py --symbol NABIL --ollama --ollama-model llama3
```

### Fast Analysis (1-year data, quick turnaround)
```bash
python nepse_live.py --symbol SNLI --fast
```

---

## 🧩 EnhancedModel (v2.1) — Modular “Feature Plugins” + Decision Engine

If you want to experiment with **leading indicators**, **smart-money context**, **index dependency**, and **news/sector sentiment** *without editing the core ensemble*, use `EnhancedModel`.

- **What it is**: a small registry that lets you add/compute extra signals from any dict-like context.
- **What it’s not**: it does not replace the ML ensemble in `models.py`; it’s meant to *compose* with your existing pipeline.

### Run the demo

```bash
python enhanced_model_demo.py
```

### Typical integration point

Use it after you’ve already computed your normal pipeline outputs (OHLCV features, floorsheet smart money, NEPSE index features, sentiment score). Package those into a `context` dict and call:

- `EnhancedModel.compute_features(context)`
- `EnhancedModel.calculate_confidence(...)`
- `EnhancedModel.decision_engine(...)`

## ✅ Verification (v6.1/v6.2)

End-to-end verification was run in **Fast mode** (no Ollama required):

- **Command**: `python nepse_live.py --symbol SNLI --fast`
- **Status**: Exit 0, ML training succeeded (LightGBM may be skipped automatically if unsupported on your local build)
- **Example metrics (SNLI, 1y, 2026-03-26)**:
  - **Avg directional accuracy**: ~56.0%
  - **Avg MAE**: ~27,942.12 (note: fold metrics can vary by data/source)
  - **Forecast horizon**: 7 Nepal trading sessions

---

## 📊 Output Explained

-   **Terminal**: Regime Phase, Trap Index, Broker HHI, ATR-based SL/TP, and volatility-adjusted position weight.
-   **Reports (`/reports`)**: JSON with full v6.1 metrics — probabilistic bands, divergence scores, and anomalies.
-   **Logs (`predictions_log-DATE.json`)**: Rolling prediction accuracy tracking.

---

## ⚠️ Disclaimer
For **educational and research purposes only**. Stock market predictions are probabilistic and carry inherent risk. Always use stop-losses. This is not financial advice.

**Directional Accuracy Target**: 70–75% (v6.1 Quantum Advanced Analytics Engine).
