# NEPSE Predictor

Machine-learning pipeline for the Nepal Stock Exchange (NEPSE): historical fetch, feature engineering, regime-aware ensemble models, optional floorsheet “smart money” signals, and JSON reports. Intended for **research and education**, not trading advice.

## What it does

- Pulls price history and related context, builds technical and calendar-aware features, and runs an ensemble (LightGBM, XGBoost, scikit-learn) with optional Optuna tuning.
- Detects market regime, flags anomalies (including news-linked context when headlines are available), and suggests strategy-style outputs with ATR-based risk framing.
- Optional **Ollama** integration for headline sentiment and an AI narrative summary (`--ollama`).
- Optional **EnhancedModel** path for experimenting with extra signals via a small plugin-style API (`enhanced_model_demo.py`).

## Feature highlights (by version)

**Core (v6.0)**  
Smart-money style floorsheet analysis (broker concentration / HHI, trap-style scores), regime detection (`BULL` / `BEAR` / `SIDEWAYS` / `MANIPULATION`), behavioral features (e.g. FOMO / panic-style indices), and a Sharpe-oriented ensemble objective.

**Advanced analytics (v6.1)**  
GARCH-style volatility features, price–volume divergence, ATR-based stop/take-profit distances, volatility-aware sizing context, and structured NLP sentiment when Ollama is enabled.

**Resilience and reporting (v6.2)**  
Stronger handling around model failures and missing Ollama, anomaly detection correlated with news categories when available, and clearer surfacing of news-linked anomalies in live output.

**Adaptive signal layer (current)**  
Multi-step entry confirmation, explicit regime-aware trading behavior, volatility-adjusted sizing, drawdown-aware execution throttling, richer trade diagnostics, and buy-and-hold alpha benchmarking in the ML backtest.

## Requirements

- Python 3.10+ recommended  
- Dependencies: see `requirements.txt` (pandas, numpy, scikit-learn, lightgbm, xgboost, optuna, matplotlib, colorama, tabulate, requests, lxml, beautifulsoup4).

### Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate with `source .venv/bin/activate`.

### Optional: Ollama (sentiment + AI summary)

```bash
ollama pull llama3
```

If Ollama is not running, the pipeline is designed to fall back without blocking the rest of the run.

## How to run

### Interactive menu

```bash
python nepse_live.py
```

### Single symbol (main CLI)

| Flag | Purpose |
|------|--------|
| `--symbol SYM` | Ticker (e.g. `NABIL`, `NTC`) |
| `--predict N` | Forecast horizon in Nepal trading sessions (default `7`, typically 5–10) |
| `--years N` | Years of history (default `2`) |
| `--backtest` | Include walk-forward backtest |
| `--fast` | Shorter history / lighter run (about one year, less heavy optimization) |
| `--no-ml` | Skip ML; statistical path only |
| `--ollama` | Enable Ollama sentiment + analyst summary |
| `--ollama-model NAME` | Model name (default `llama3`) |
| `--seed N` | Reproducibility seed (default `42`) |
| `--debug` | Extra diagnostic output |
| `--list` | Print symbols and exit |

Examples:

```bash
python nepse_live.py --symbol NABIL --predict 7
python nepse_live.py --symbol NABIL --years 5 --backtest
python nepse_live.py --symbol NABIL --ollama --ollama-model llama3
python nepse_live.py --symbol SNLI --fast
python nepse_live.py --symbol NIFRA --no-ml
```

### Batch predictions (`run_all.py`)

Runs many symbols in parallel and prints JSON; also writes `predictions_log-YYYY-MM-DD.json`.

```bash
python run_all.py
python run_all.py --symbols NABIL NTC
python run_all.py --workers 8 --no-ml
```

Default worker count is `8` (adjust with `--workers`).

### EnhancedModel demo

```bash
python enhanced_model_demo.py
```

Use this to experiment with `EnhancedModel.compute_features`, `calculate_confidence`, and `decision_engine` without changing the core ensemble in `models.py`.

## Outputs

- **`reports/`** — Per-run JSON reports (metrics, bands, anomalies, etc.).
- **`predictions_log-YYYY-MM-DD.json`** — Rolling log of predicted vs actual closes when you use batch or repeated runs.
- **Terminal** — Regime, smart-money summaries, SL/TP style levels, optional Ollama block.
- **ML backtest** — Strategy return, drawdown, exposure, buy-and-hold benchmark comparison, and trade-level diagnostics such as signal strength, regime, volatility, and entry/exit reasons.

## Calendar and data

Trading-date logic uses Nepal-market calendar helpers (`nepse_market_calendar.py`, `nepal_calendar.py`, holiday CSVs such as `nepse_holidays.csv`). Symbol lists may be cached (e.g. `symbols_cache.json`) to limit repeated API calls.

## Tests

```bash
python test_calendar_and_determinism.py
```

Other small scripts (`test_nepse_live.py`, `test_live.py`) are ad hoc checks; use them if you are validating fetch or live paths on your machine.

## Verification note

End-to-end checks have been run in **fast** mode (e.g. `python nepse_live.py --symbol SNLI --fast`). Reported directional accuracy and MAE **depend on symbol, date range, and data source**; treat any single run as illustrative, not a guarantee of future performance.

## Disclaimer

This project is for **education and research only**. Markets are risky; predictions are uncertain. This is **not financial advice**. Always do your own due diligence and risk management.
