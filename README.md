## NEPSE prediction system (Python)

This project builds a **daily NEPSE (Nepal Stock Exchange) prediction pipeline**:
- Fetches **live + historical** OHLCV data (via `merolagani` / `nepalstock` / `sharesansar` fallbacks)
- Maintains a **growing per-symbol CSV dataset** in `data/`
- Cleans/dedupes and generates features (moving averages, returns, volume trends, volatility)
- Trains **Linear Regression** (baseline) + **Random Forest** (better)
- Forecasts next **5–10 business days**
- Optionally calls **local Ollama** (mistral recommended) to explain trend + signals + risks
- Saves outputs into `outputs/` (report `.txt` + `.json` + forecast plot `.png`)

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run (one symbol)

```bash
python nepse_system.py --symbol NABIL --years 5 --horizon 7
```

If you don’t have Ollama running (or you want to skip it):

```bash
python nepse_system.py --symbol NABIL --no-ollama
```

### Ollama integration

1. Install Ollama and pull a model:

```bash
ollama pull mistral
```

2. Run the daily script (it will call `http://localhost:11434/api/generate`):

```bash
python nepse_system.py --symbol NABIL --ollama-model mistral
```

### Output locations

- **Dataset (growing)**: `data/<SYMBOL>.csv`
- **Reports**: `outputs/<SYMBOL>_<timestamp>.txt` and `.json`
- **Plot**: `outputs/<SYMBOL>_<timestamp>.png`

### Daily automation (cron)

Run every day at 6:15 PM (example):

```bash
crontab -e
```

Add (update paths for your machine):

```bash
15 18 * * * cd /Users/manishkumarsingh/py/nepse_predict && /Users/manishkumarsingh/py/nepse_predict/.venv/bin/python nepse_system.py --symbol NABIL --years 5 --horizon 7 >> outputs/cron.log 2>&1
```

### Notes / realism

- Models are trained with a **time-based holdout** (no random shuffling).
- Forecast is **recursive** (each predicted day feeds the next day’s features).
- A **±10% daily circuit filter** is applied as a guardrail.
- Nepal public holidays are not encoded; business days are approximated as **Mon–Fri**.

# nepse_predict
